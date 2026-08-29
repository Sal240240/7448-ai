"""
Phase 1 baseline: deep MLP (spec section 5.1), trained end-to-end on all
targets jointly with a masked loss, and benchmarked against the per-target
elastic net baseline (train_elastic_net.py) on the same splits -- per spec:
"if you don't beat the 2019 linear baseline, the deep model isn't earning
its complexity yet."

Features (taxonomy) and targets (log1p metabolite abundance) are both
z-scored using train-set statistics before training; target normalization
is computed per-column over only that column's *measured* entries (see
standardize_metabolites.py for why NaN != 0 here).

Output:
    experiments/mlp_baseline_metrics.csv   (per-target r/MAE on val + test, same schema as the elastic net CSV)
    experiments/mlp_training_log.csv       (per-epoch train/val loss)

Usage:
    python scripts/train_mlp.py --epochs 200 --patience 15
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import Dataset, pearson_per_column  # noqa: E402
from models.mlp_baseline import MLPBaseline, masked_mse_loss  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def to_tensors(X: pd.DataFrame, Y: pd.DataFrame, M: pd.DataFrame, x_mean, x_std, y_mean, y_std):
    Xn = ((X.values - x_mean) / x_std).astype("float32")
    Yn = ((Y.fillna(0).values - y_mean) / y_std).astype("float32")
    Mv = M.values.astype("float32")
    return torch.from_numpy(Xn), torch.from_numpy(Yn), torch.from_numpy(Mv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = Dataset()
    X_train, Y_train, M_train = ds.xy("train")
    X_val, Y_val, M_val = ds.xy("val")
    X_test, Y_test, M_test = ds.xy("test")

    x_mean, x_std = X_train.values.mean(axis=0), X_train.values.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = np.nanmean(Y_train.where(M_train).values, axis=0)
    y_std = np.nanstd(Y_train.where(M_train).values, axis=0)
    y_mean = np.nan_to_num(y_mean, nan=0.0)
    y_std = np.nan_to_num(y_std, nan=1.0)
    y_std[y_std == 0] = 1.0

    Xtr, Ytr, Mtr = to_tensors(X_train, Y_train, M_train, x_mean, x_std, y_mean, y_std)
    Xva, Yva, Mva = to_tensors(X_val, Y_val, M_val, x_mean, x_std, y_mean, y_std)
    Xte, Yte, Mte = to_tensors(X_test, Y_test, M_test, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr, Mtr), batch_size=args.batch_size, shuffle=True)

    model = MLPBaseline(n_features=ds.n_features, n_targets=ds.n_targets, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Features: {ds.n_features}, targets: {ds.n_targets}")
    print(f"Train/val/test samples: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    log_rows = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mse_loss(pred, yb, mb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = masked_mse_loss(val_pred, Yva, Mva).item()

        train_loss = float(np.mean(train_losses))
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    print(f"\nTraining took {time.time() - t0:.1f}s, best val_loss={best_val_loss:.4f}")
    model.load_state_dict(best_state)

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(log_rows).to_csv(EXPERIMENTS_DIR / "mlp_training_log.csv", index=False)

    model.eval()
    with torch.no_grad():
        val_pred_n = model(Xva).numpy()
        test_pred_n = model(Xte).numpy()

    # un-normalize predictions and targets back to log1p-abundance scale for interpretable metrics
    val_pred = val_pred_n * y_std + y_mean
    test_pred = test_pred_n * y_std + y_mean
    val_true = Y_val.values
    test_true = Y_test.values

    r_val = pearson_per_column(val_true, val_pred, M_val.values)
    r_test = pearson_per_column(test_true, test_pred, M_test.values)

    mae_val = np.full(ds.n_targets, np.nan)
    mae_test = np.full(ds.n_targets, np.nan)
    for j in range(ds.n_targets):
        mv, mt = M_val.values[:, j], M_test.values[:, j]
        if mv.sum() >= 3:
            mae_val[j] = np.mean(np.abs(val_true[mv, j] - val_pred[mv, j]))
        if mt.sum() >= 3:
            mae_test[j] = np.mean(np.abs(test_true[mt, j] - test_pred[mt, j]))

    results = pd.DataFrame({
        "target": ds.Y.columns,
        "n_train": M_train.sum().values,
        "n_val": M_val.sum().values,
        "n_test": M_test.sum().values,
        "pearson_val": r_val,
        "pearson_test": r_test,
        "mae_val": mae_val,
        "mae_test": mae_test,
    })
    out = EXPERIMENTS_DIR / "mlp_baseline_metrics.csv"
    results.to_csv(out, index=False)

    print(f"\n-> {out}")
    print(f"Median val Pearson r:  {results['pearson_val'].median():.3f}  (n={results['pearson_val'].notna().sum()})")
    print(f"Median test Pearson r: {results['pearson_test'].median():.3f}  (n={results['pearson_test'].notna().sum()})")
    print(f"Mean val Pearson r:    {results['pearson_val'].mean():.3f}")
    print(f"Mean test Pearson r:   {results['pearson_test'].mean():.3f}")


if __name__ == "__main__":
    main()
