#!/usr/bin/env python3
"""
PubMed Central Review Downloader
---------------------------------
1. Searches PubMed for a given term
2. Filters for Review / Systematic Review publication types
3. Filters by minimum citation count (via iCite)
4. Downloads full-text PDFs for open-access articles via PMC OA Web Service
5. Saves a TSV summary of all results

Usage:
    python pubmed_review_downloader.py \
        --query "tuberculosis drug resistance" \
        --max-results 500 \
        --min-citations 20 \
        --output-dir ./reviews \
        --email your@email.com

Requirements:
    pip install requests
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ── NCBI E-utilities ──────────────────────────────────────────────────────────
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA  = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
ICITE   = "https://icite.od.nih.gov/api/pubs"
IDCONV  = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

REVIEW_TYPES = {"Review", "Systematic Review", "Meta-Analysis"}

# ---------------------------------------------------------------------------

def ncbi_get(endpoint: str, params: dict, retries: int = 3) -> requests.Response:
    """GET wrapper with retry and rate limiting."""
    for attempt in range(retries):
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(2 ** attempt)


def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Step 1: Search ────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int, email: str, api_key: str) -> list[str]:
    print(f"\n[1] Searching PubMed: '{query}' (max {max_results})...")
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = ncbi_get(f"{EUTILS}/esearch.fcgi", params)
    data = r.json()
    pmids = data["esearchresult"]["idlist"]
    total = int(data["esearchresult"]["count"])
    print(f"  Found {total} total results; retrieved {len(pmids)} PMIDs.")
    return pmids


# ── Step 2: Fetch details and filter for reviews ──────────────────────────────

def fetch_article_details(pmids: list[str], email: str, api_key: str) -> list[dict]:
    """Fetch PubMed records in batches; return list of article dicts."""
    print(f"\n[2] Fetching article details for {len(pmids)} records...")
    articles = []
    batch_size = 200

    for i, batch in enumerate(chunked(pmids, batch_size)):
        print(f"  Batch {i + 1}/{-(-len(pmids) // batch_size)}: {len(batch)} records", end="\r")
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
            "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        r = ncbi_get(f"{EUTILS}/efetch.fcgi", params)
        root = ET.fromstring(r.text)

        for article_elem in root.findall(".//PubmedArticle"):
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Publication types
            pub_types = {
                pt.text
                for pt in article_elem.findall(".//PublicationType")
                if pt.text
            }

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()) if title_elem is not None else ""

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Year
            year = (
                article_elem.findtext(".//PubDate/Year")
                or article_elem.findtext(".//PubDate/MedlineDate", "")[:4]
                or ""
            )

            # Journal
            journal = article_elem.findtext(".//Journal/Title", "") or \
                      article_elem.findtext(".//MedlineTA", "")

            # DOI
            doi = ""
            for id_elem in article_elem.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text or ""
                    break

            # PMC ID
            pmc_id = ""
            for id_elem in article_elem.findall(".//ArticleId"):
                if id_elem.get("IdType") == "pmc":
                    pmc_id = id_elem.text or ""
                    break

            articles.append({
                "pmid": pmid,
                "pmc_id": pmc_id,
                "title": title.strip(),
                "authors": author_str,
                "year": year,
                "journal": journal,
                "doi": doi,
                "pub_types": pub_types,
                "citations": 0,          # filled in Step 3
                "oa_license": "",        # filled in Step 4
                "oa_pdf_url": "",        # filled in Step 4
                "downloaded": False,
            })

        time.sleep(0.34)  # ~3 req/s rate limit (10/s with API key)

    print(f"  Fetched {len(articles)} article records.")
    return articles


def filter_reviews(articles: list[dict]) -> list[dict]:
    reviews = [a for a in articles if a["pub_types"] & REVIEW_TYPES]
    print(f"\n  Filtered to {len(reviews)} reviews / systematic reviews / meta-analyses.")
    return reviews


# ── Step 3: Citation counts via iCite ─────────────────────────────────────────

def add_citation_counts(articles: list[dict]) -> None:
    print(f"\n[3] Fetching citation counts from iCite for {len(articles)} articles...")
    pmids = [a["pmid"] for a in articles if a["pmid"]]
    counts: dict[str, int] = {}

    for batch in chunked(pmids, 100):
        r = requests.get(ICITE, params={"pmids": ",".join(batch)}, timeout=30)
        r.raise_for_status()
        for pub in r.json().get("data", []):
            counts[str(pub["pmid"])] = pub.get("citation_count", 0)
        time.sleep(0.2)

    for a in articles:
        a["citations"] = counts.get(a["pmid"], 0)

    print(f"  Citation counts retrieved.")


def filter_by_citations(articles: list[dict], min_citations: int) -> list[dict]:
    filtered = [a for a in articles if a["citations"] >= min_citations]
    print(f"  Filtered to {len(filtered)} reviews with ≥ {min_citations} citations.")
    return filtered


# ── Step 4: Resolve PMC IDs and download full text ────────────────────────────

def resolve_pmc_ids(articles: list[dict], email: str) -> None:
    """Fill in missing pmc_id using NCBI ID converter."""
    missing = [a for a in articles if not a["pmc_id"] and a["pmid"]]
    if not missing:
        return
    print(f"\n  Resolving PMC IDs for {len(missing)} articles...")
    for batch in chunked(missing, 50):
        pmids = [a["pmid"] for a in batch]
        r = requests.get(
            IDCONV,
            params={"ids": ",".join(pmids), "format": "json", "email": email},
            timeout=30,
        )
        if not r.ok:
            continue
        data = r.json()
        id_map = {rec.get("pmid", ""): rec.get("pmcid", "") for rec in data.get("records", [])}
        for a in batch:
            a["pmc_id"] = id_map.get(a["pmid"], "").replace("PMC", "")
        time.sleep(0.3)


def is_commercial_license(license: str) -> bool:
    """
    Returns True if the license permits commercial use.
    CC BY, CC BY-SA, CC BY-ND  → commercial OK
    CC BY-NC, CC BY-NC-SA, CC BY-NC-ND → non-commercial only
    NO-CC / blank → unknown; treated as non-commercial to be safe
    """
    if not license:
        return False
    lic = license.upper()
    return "CC" in lic and "NC" not in lic


def get_oa_info(pmc_id: str) -> tuple[str, str]:
    """
    Query PMC OA Web Service.
    Returns (pdf_url, license_string). Both empty if not open access.
    """
    if not pmc_id:
        return "", ""
    r = requests.get(PMC_OA, params={"id": f"PMC{pmc_id}"}, timeout=20)
    if not r.ok:
        return "", ""
    try:
        root = ET.fromstring(r.text)
        if root.find(".//error") is not None:
            return "", ""
        record = root.find(".//record")
        if record is None:
            return "", ""
        license_str = record.get("license", "")
        link = record.find(".//link[@format='pdf']")
        if link is None:
            return "", license_str
        href = link.get("href", "")
        if href.startswith("ftp://"):
            href = href.replace("ftp://", "https://", 1)
        return href, license_str
    except ET.ParseError:
        return "", ""


def download_pdfs(articles: list[dict], output_dir: Path, oa_comm_only: bool = False) -> None:
    flag = " (commercial licenses only)" if oa_comm_only else ""
    print(f"\n[4] Downloading full-text PDFs{flag} to '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    oa_count = 0
    for i, a in enumerate(articles):
        label = f"  [{i+1}/{len(articles)}] PMID {a['pmid']}"
        if not a["pmc_id"]:
            print(f"{label} — no PMC ID, skipping.")
            continue

        pdf_url, license_str = get_oa_info(a["pmc_id"])
        a["oa_license"] = license_str

        if not pdf_url:
            print(f"{label} — not open access, skipping.")
            time.sleep(0.2)
            continue

        if oa_comm_only and not is_commercial_license(license_str):
            print(f"{label} — license '{license_str or 'unknown'}' is non-commercial, skipping.")
            time.sleep(0.2)
            continue

        a["oa_pdf_url"] = pdf_url
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in a["title"])[:60]
        filename = output_dir / f"PMC{a['pmc_id']}_{safe_title}.pdf"

        if filename.exists():
            print(f"{label} — already downloaded.")
            a["downloaded"] = True
            oa_count += 1
            continue

        try:
            r = requests.get(pdf_url, timeout=60, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{label} — [{license_str or 'OA'}] saved '{filename.name}'")
            a["downloaded"] = True
            oa_count += 1
        except Exception as e:
            print(f"{label} — download failed: {e}")

        time.sleep(0.5)

    print(f"\n  Downloaded {oa_count} / {len(articles)} PDFs.")


# ── Step 5: Save summary TSV ──────────────────────────────────────────────────

def save_summary(articles: list[dict], output_dir: Path) -> None:
    tsv_path = output_dir / "summary.tsv"
    headers = ["PMID", "PMC_ID", "Year", "Citations", "OA_License", "Commercial", "Downloaded", "Title", "Authors", "Journal", "DOI", "OA_PDF_URL"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for a in sorted(articles, key=lambda x: -x["citations"]):
            comm = "yes" if is_commercial_license(a["oa_license"]) else ("no" if a["oa_license"] else "")
            row = [
                a["pmid"], a["pmc_id"], a["year"], str(a["citations"]),
                a["oa_license"], comm,
                "yes" if a["downloaded"] else "no",
                a["title"], a["authors"], a["journal"], a["doi"], a["oa_pdf_url"],
            ]
            f.write("\t".join(row) + "\n")
    print(f"\n  Summary saved → {tsv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search PubMed for reviews, filter by citations, download PDFs."
    )
    parser.add_argument("--query",         required=True,  help='Search term, e.g. "tuberculosis drug resistance"')
    parser.add_argument("--max-results",   type=int, default=500, help="Max PubMed records to retrieve (default: 500)")
    parser.add_argument("--min-citations", type=int, default=10,  help="Minimum citation count to include (default: 10)")
    parser.add_argument("--output-dir",    default="./reviews",   help="Folder to save PDFs and summary (default: ./reviews)")
    parser.add_argument("--email",         required=True,  help="Your email (required by NCBI)")
    parser.add_argument("--api-key",       default="",     help="NCBI API key (optional; raises rate limit from 3 to 10 req/s)")
    parser.add_argument("--no-download",   action="store_true",   help="Skip PDF download; only produce the summary TSV")
    parser.add_argument("--oa-comm-only",  action="store_true",   help="Only download PDFs with a commercial-friendly OA license (CC BY, CC BY-SA, CC BY-ND)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 1. Search
    pmids = search_pubmed(args.query, args.max_results, args.email, args.api_key)
    if not pmids:
        print("No results. Exiting.")
        sys.exit(0)

    # 2. Fetch details + filter reviews
    articles = fetch_article_details(pmids, args.email, args.api_key)
    reviews = filter_reviews(articles)
    if not reviews:
        print("No reviews found. Try a broader query.")
        sys.exit(0)

    # 3. Citation counts + filter
    add_citation_counts(reviews)
    filtered = filter_by_citations(reviews, args.min_citations)
    if not filtered:
        print(f"No reviews with ≥ {args.min_citations} citations. Try lowering --min-citations.")
        sys.exit(0)

    # 4. Download PDFs
    if not args.no_download:
        resolve_pmc_ids(filtered, args.email)
        download_pdfs(filtered, output_dir, oa_comm_only=args.oa_comm_only)
    else:
        print("\n[4] Skipping PDF download (--no-download).")
        output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Summary
    save_summary(filtered, output_dir)

    print(f"\nDone. {len(filtered)} reviews meet your criteria.")
    if not args.no_download:
        downloaded = sum(1 for a in filtered if a["downloaded"])
        print(f"  {downloaded} PDFs downloaded (rest are not open access).")


if __name__ == "__main__":
    main()
