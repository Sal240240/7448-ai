"""
Phase 2 step 4: resolve the PubMed IDs behind each association into real
citations.

Disbiome hands us PubMed IDs; a bare "PMID 24997786" is not a citation a
reader can evaluate. This resolves them to title / journal / year / authors
through NCBI's E-utilities, which is the documented, free, intended way to
query PubMed programmatically (no scraping of pubmed.ncbi.nlm.nih.gov pages).

The point is accountability: every microbe-disease claim the webapp shows
should be traceable to a specific paper the reader can go read, including its
year -- a 2011 association and a 2023 association carry different weight, and
hiding that behind "studies show" is how health tools lose the right to be
trusted.

NCBI asks for <=3 requests/second without an API key and for a tool/email
identifier on each request; both are honored below. Responses are cached under
data/external/pubmed/.

Output:
    data/reference/publications.csv   (checked in; one row per PMID)

Usage:
    python scripts/fetch_pubmed.py
    python scripts/fetch_pubmed.py --max-pmids 500
"""
import argparse
import json
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from _paths import EXTERNAL_DIR, REFERENCE_DIR

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 100          # E-utilities handles a few hundred IDs per POST comfortably
REQUEST_DELAY_S = 0.4     # <3 req/s, NCBI's documented unauthenticated ceiling
TOOL_NAME = "7448-ai"
CACHE_PATH = EXTERNAL_DIR / "pubmed" / "records.json"


def parse_article(article: ET.Element) -> dict | None:
    pmid_el = article.find(".//PMID")
    if pmid_el is None:
        return None

    title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
    journal_el = article.find(".//Journal/ISOAbbreviation")
    if journal_el is None:
        journal_el = article.find(".//Journal/Title")

    year = ""
    for path in (".//JournalIssue/PubDate/Year", ".//JournalIssue/PubDate/MedlineDate", ".//ArticleDate/Year"):
        el = article.find(path)
        if el is not None and el.text:
            year = el.text[:4]
            break

    authors = []
    for author in article.findall(".//Author")[:3]:
        last = author.find("LastName")
        if last is not None and last.text:
            authors.append(last.text)

    return {
        "pmid": pmid_el.text,
        "title": title.strip(),
        "journal": (journal_el.text if journal_el is not None else "").strip(),
        "year": year,
        # Singular, matching the column fetch_disbiome.py writes and
        # src/simulate.py reads. A plural spelling here silently produced blank
        # authors on every citation in the UI.
        "first_author": ", ".join(authors),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/",
    }


def fetch_pmids(session: requests.Session, pmids: list[str]) -> list[dict]:
    resp = session.post(
        EFETCH,
        data={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": TOOL_NAME},
        timeout=120,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    records = [parse_article(a) for a in root.findall(".//PubmedArticle")]
    time.sleep(REQUEST_DELAY_S)
    return [r for r in records if r]


def collect_pmids() -> list[str]:
    assoc_path = REFERENCE_DIR / "taxon_disease_associations.csv"
    if not assoc_path.exists():
        return []
    assoc = pd.read_csv(assoc_path)
    pmids = (
        assoc["pubmed_ids"].fillna("").astype(str).str.split(";").explode().str.strip()
    )
    return sorted({p for p in pmids if p.isdigit()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-pmids", type=int, default=None, help="Debug: cap how many PMIDs to resolve")
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    pmids = collect_pmids()
    if not pmids:
        print("No PubMed IDs found -- run fetch_disbiome.py first.")
        return
    if args.max_pmids:
        pmids = pmids[: args.max_pmids]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if CACHE_PATH.exists() and not args.force:
        cached = {r["pmid"]: r for r in json.loads(CACHE_PATH.read_text(encoding="utf-8"))}

    todo = [p for p in pmids if p not in cached]
    print(f"{len(pmids)} PubMed IDs referenced by the association table; {len(cached)} cached, {len(todo)} to fetch")

    session = requests.Session()
    session.headers.update({"User-Agent": f"{TOOL_NAME}-research-pipeline"})

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        try:
            for record in fetch_pmids(session, batch):
                cached[record["pmid"]] = record
        except (requests.RequestException, ET.ParseError) as e:
            print(f"  ! batch at {i} failed ({e}), continuing")
            continue
        print(f"  resolved {min(i + BATCH_SIZE, len(todo))}/{len(todo)}")

    if not cached:
        print("Nothing resolved.")
        return

    CACHE_PATH.write_text(json.dumps(list(cached.values())), encoding="utf-8")

    pubs = pd.DataFrame(cached.values()).sort_values("pmid").reset_index(drop=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / "publications.csv"

    # Enrich rather than overwrite. fetch_disbiome.py owns this file: it carries
    # publication_id and the per-study methodological-quality flags, which NCBI
    # does not provide and which the outcome layer weights associations by.
    # Writing this file wholesale from here would silently drop those columns.
    # What PubMed adds is canonical titles/journals where Disbiome's are thin.
    if out.exists():
        existing = pd.read_csv(out, dtype={"pmid": str})
        enrich = pubs.set_index("pmid")
        merged = existing.copy()
        for column in ["title", "journal", "year", "first_author"]:
            if column not in merged.columns or column not in enrich.columns:
                continue
            filled = merged["pmid"].map(enrich[column])
            blank = merged[column].isna() | merged[column].astype(str).str.strip().eq("")
            merged.loc[blank, column] = filled[blank]
        if "url" in pubs.columns:
            merged["url"] = merged["pmid"].map(enrich["url"]).fillna("")
        merged.to_csv(out, index=False)
        print(f"   enriched {int(existing['pmid'].isin(pubs['pmid']).sum())} existing rows "
              f"(publication_id and quality flags preserved)")
    else:
        pubs.to_csv(out, index=False)

    print(f"\n-> {out}  ({len(pubs)} publications)")
    resolved_years = pd.to_numeric(pubs["year"], errors="coerce").dropna()
    if len(resolved_years):
        print(f"   median year: {int(resolved_years.median())}  "
              f"(range {int(resolved_years.min())}-{int(resolved_years.max())})")
    print("\nMost-cited journals:")
    print(pubs[pubs["journal"] != ""]["journal"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
