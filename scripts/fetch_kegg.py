"""
Phase 2 step 2: annotate metabolites with KEGG pathways, chemical class, and
disease links.

Why KEGG and not HMDB: the modeling targets are HMDB IDs, and HMDB's own
metabolite pages carry disease associations -- but hmdb.ca returns 403 to
scripted requests, and working around a site's bot protection isn't something
this pipeline does. The Borenstein annotation files already carry a KEGG
compound ID for 969 of our 1,384 metabolites, and KEGG publishes a documented
REST API that is free for academic use. So the same enrichment (what pathway
is this metabolite in, what class of compound is it, what diseases is it
linked to) is reachable through the front door.

Coverage is reported honestly at the end: ~70% of targets carry a KEGG ID at
all, and not every KEGG compound has pathway or disease links, so the outcome
layer downstream has to tolerate gaps rather than assume full annotation.

Raw responses are cached under data/external/kegg/ so re-runs cost nothing and
the API isn't hit repeatedly for the same compound.

Output:
    data/reference/metabolite_kegg.csv   (checked in; one row per annotated KEGG compound)

Usage:
    python scripts/fetch_kegg.py
    python scripts/fetch_kegg.py --force     # ignore cache, re-fetch everything
"""
import argparse
import time

import pandas as pd
import requests

from _paths import EXTERNAL_DIR, REFERENCE_DIR

KEGG_GET = "https://rest.kegg.jp/get/"
# KEGG's REST API accepts up to 10 entries per request; batching keeps this to
# ~100 calls instead of ~1000.
BATCH_SIZE = 10
REQUEST_DELAY_S = 0.34  # ~3 req/s, well inside KEGG's fair-use expectations
CACHE_DIR = EXTERNAL_DIR / "kegg"


def parse_kegg_record(text: str) -> dict:
    """Parse one KEGG flat-file record into the fields we care about.

    KEGG's format is column-oriented: a field name in columns 0-11, continuation
    lines indented. BRITE is a nested tree, but the useful signal for us is the
    leaf hierarchy under 'Compounds with biological roles' -- we keep the whole
    indented block's distinct category lines and let the caller pick.
    """
    fields: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.startswith("///"):
            continue
        if not line.startswith(" "):
            current = line[:12].strip()
            content = line[12:].strip()
            fields.setdefault(current, [])
            if content:
                fields[current].append(content)
        elif current:
            fields[current].append(line.strip())

    entry_id = fields.get("ENTRY", [""])[0].split()[0] if fields.get("ENTRY") else ""
    names = [n.rstrip(";") for n in fields.get("NAME", [])]

    pathways = []
    for line in fields.get("PATHWAY", []):
        parts = line.split(None, 1)
        if len(parts) == 2:
            pathways.append(parts[1])

    # BRITE lines that name a hierarchy root look like "Lipids [BR:br08002]";
    # the indented leaves below are the actual classification.
    brite_roots = [ln.split(" [BR:")[0].strip() for ln in fields.get("BRITE", []) if " [BR:" in ln]

    diseases = []
    for line in fields.get("DISEASE", []):
        parts = line.split(None, 1)
        diseases.append(parts[1] if len(parts) == 2 else line)

    return {
        "kegg_id": entry_id,
        "kegg_name": names[0] if names else "",
        "kegg_synonyms": "; ".join(names[1:]) if len(names) > 1 else "",
        "formula": fields.get("FORMULA", [""])[0] if fields.get("FORMULA") else "",
        "pathways": "; ".join(pathways),
        "n_pathways": len(pathways),
        "brite_classes": "; ".join(dict.fromkeys(brite_roots)),
        "kegg_diseases": "; ".join(diseases),
    }


def fetch_batch(session: requests.Session, kegg_ids: list[str], force: bool) -> list[str]:
    """Return raw record text per compound, using the on-disk cache where possible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results, to_fetch = {}, []
    for kid in kegg_ids:
        cache_path = CACHE_DIR / f"{kid}.txt"
        if cache_path.exists() and not force:
            results[kid] = cache_path.read_text(encoding="utf-8")
        else:
            to_fetch.append(kid)

    if to_fetch:
        resp = session.get(KEGG_GET + "+".join(to_fetch), timeout=60)
        resp.raise_for_status()
        # A multi-entry response is records concatenated with "///" separators,
        # in request order; entries KEGG doesn't know are simply absent, so we
        # match on the ENTRY line rather than assuming positional alignment.
        for chunk in resp.text.split("///"):
            if not chunk.strip():
                continue
            entry = chunk.strip().split(None, 2)
            if len(entry) >= 2 and entry[0] == "ENTRY":
                kid = entry[1]
                results[kid] = chunk
                (CACHE_DIR / f"{kid}.txt").write_text(chunk, encoding="utf-8")
        time.sleep(REQUEST_DELAY_S)

    return [results[k] for k in kegg_ids if k in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Ignore the cache and re-fetch every compound")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only fetch the first N compounds")
    args = parser.parse_args()

    ref_path = REFERENCE_DIR / "metabolite_reference.csv"
    if not ref_path.exists():
        print(f"{ref_path} not found -- run build_metabolite_reference.py first.")
        return

    ref = pd.read_csv(ref_path)
    kegg_ids = sorted({k for k in ref["kegg_id"].dropna().astype(str) if k.startswith("C")})
    if args.limit:
        kegg_ids = kegg_ids[: args.limit]

    print(f"{len(ref)} metabolites in reference, {len(kegg_ids)} with a KEGG compound ID")

    session = requests.Session()
    session.headers.update({"User-Agent": "7448-ai-research-pipeline"})

    records = []
    for i in range(0, len(kegg_ids), BATCH_SIZE):
        batch = kegg_ids[i : i + BATCH_SIZE]
        try:
            for text in fetch_batch(session, batch, args.force):
                records.append(parse_kegg_record(text))
        except requests.RequestException as e:
            print(f"  ! batch {i // BATCH_SIZE} failed ({e}), continuing")
            continue
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {min(i + BATCH_SIZE, len(kegg_ids)):4d}/{len(kegg_ids)} compounds")

    if not records:
        print("No KEGG records retrieved.")
        return

    kegg = pd.DataFrame(records).drop_duplicates(subset="kegg_id").sort_values("kegg_id")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / "metabolite_kegg.csv"
    kegg.to_csv(out, index=False)

    merged = ref.merge(kegg, on="kegg_id", how="left")
    n_targets = int(ref["in_model_targets"].sum()) if "in_model_targets" in ref else len(ref)
    # Coverage is per modeling target, not per KEGG compound -- several HMDB IDs
    # can share one KEGG compound, so these differ.
    n_with_kegg = int(merged["kegg_id"].astype(str).str.startswith("C").sum())

    print(f"\n-> {out}  ({len(kegg)} KEGG compounds)")
    print(f"\nCoverage against the {n_targets} modeling targets:")
    print(f"  has KEGG ID:        {n_with_kegg:5d}  ({n_with_kegg / n_targets:.0%})")
    print(f"  has >=1 pathway:    {(merged['n_pathways'] > 0).sum():5d}  ({(merged['n_pathways'] > 0).sum() / n_targets:.0%})")
    print(f"  has BRITE class:    {merged['brite_classes'].fillna('').ne('').sum():5d}")
    print(f"  has KEGG disease:   {merged['kegg_diseases'].fillna('').ne('').sum():5d}")
    print("\nMost common pathways:")
    pathway_counts = (
        merged["pathways"].fillna("").str.split("; ").explode().replace("", pd.NA).dropna().value_counts()
    )
    print(pathway_counts.head(12).to_string())


if __name__ == "__main__":
    main()
