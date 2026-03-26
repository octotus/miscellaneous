#!/usr/bin/env python3
"""
PubMed Downloader
-----------------
1. Searches PubMed for a given term
2. Optionally filters for Review / Systematic Review / Meta-Analysis types (--reviews-only)
3. Filters by minimum citation count (via iCite)
4. Downloads full-text PDFs for open-access articles via PMC OA Web Service
5. Saves a TSV summary of all results

Usage:
    python pubmed_review_downloader.py \
        --query "tuberculosis drug resistance" \
        --max-results 500 \
        --min-citations 20 \
        --from-date 01-01-2015 \
        --to-date 31-12-2023 \
        --output-dir ./reviews \
        --email your@email.com

Requirements:
    pip install requests
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
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

def parse_date(date_str: str, label: str) -> str:
    """Parse DD-MM-YYYY and return YYYY/MM/DD for NCBI."""
    try:
        dt = datetime.strptime(date_str.strip(), "%d-%m-%Y")
        return dt.strftime("%Y/%m/%d")
    except ValueError:
        print(f"Error: --{label} '{date_str}' is not a valid date. Use DD-MM-YYYY format.")
        sys.exit(1)


PUBMED_MAX_PER_QUERY = 9999   # Hard NCBI limit for a single esearch call

DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _esearch_window(base_params: dict, mindate: str, maxdate: str, want: int) -> tuple[list[str], int]:
    """
    Fetch up to `want` PMIDs for a date window.
    Returns (pmid_list, total_count).
    """
    params = {
        **base_params,
        "datetype": "pdat",
        "mindate": mindate,
        "maxdate": maxdate,
        "retmax": min(want, PUBMED_MAX_PER_QUERY),
    }
    r = ncbi_get(f"{EUTILS}/esearch.fcgi", params)
    esearch = json.loads(r.text, strict=False).get("esearchresult", {})
    if "ERROR" in esearch or "error" in esearch:
        print(f"\n  NCBI error ({mindate}–{maxdate}): {esearch.get('ERROR') or esearch.get('error')}")
        return [], 0
    total = int(esearch.get("count", 0))
    return esearch.get("idlist", []), total


def _esearch_year(base_params: dict, year: int, want: int) -> list[str]:
    """
    Fetch up to `want` PMIDs for a single year.
    Automatically falls back to month-level sweep if the year exceeds
    PUBMED_MAX_PER_QUERY results. Within each month, falls back to a
    day-level sweep if the month itself exceeds the limit.
    """
    mindate = f"{year}/01/01"
    maxdate = f"{year}/12/31"

    # Check yearly count first (retmax=0 is fast)
    _, yearly_total = _esearch_window(base_params, mindate, maxdate, 0)
    time.sleep(0.11)

    if yearly_total <= PUBMED_MAX_PER_QUERY:
        pmids, _ = _esearch_window(base_params, mindate, maxdate, want)
        return pmids

    # Month-level sweep
    print(f"\n  {year}: {yearly_total} results — sweeping by month...")
    seen: set[str] = set()
    pmids: list[str] = []
    leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days = DAYS_IN_MONTH[:]
    if leap:
        days[2] = 29

    for month in range(12, 0, -1):   # December → January
        if len(pmids) >= want:
            break
        mm = f"{month:02d}"
        last_day = days[month]
        mo_min = f"{year}/{mm}/01"
        mo_max = f"{year}/{mm}/{last_day}"

        _, mo_total = _esearch_window(base_params, mo_min, mo_max, 0)
        time.sleep(0.11)

        if mo_total == 0:
            continue

        if mo_total <= PUBMED_MAX_PER_QUERY:
            batch, _ = _esearch_window(base_params, mo_min, mo_max, want - len(pmids))
            time.sleep(0.11)
            new = [p for p in batch if p not in seen]
            seen.update(new)
            pmids.extend(new)
            print(f"    {year}/{mm}: +{len(new)} ({mo_total} total)  [{len(pmids)}/{want}]", end="\r")
        else:
            # Day-level sweep for busy months
            print(f"\n    {year}/{mm}: {mo_total} results — sweeping by day...")
            for day in range(last_day, 0, -1):
                if len(pmids) >= want:
                    break
                dd = f"{day:02d}"
                date_str = f"{year}/{mm}/{dd}"
                batch, _ = _esearch_window(base_params, date_str, date_str, want - len(pmids))
                time.sleep(0.11)
                new = [p for p in batch if p not in seen]
                seen.update(new)
                pmids.extend(new)

    return pmids


FIELD_TAGS = {"all": "", "title": "[ti]", "tiab": "[tiab]"}


def search_pubmed(query: str, max_results: int, email: str, api_key: str,
                  from_date: str = "", to_date: str = "",
                  field: str = "all", db: str = "pubmed") -> list[str]:
    tag = FIELD_TAGS.get(field, "")
    term = f"{query}{tag}" if tag else query
    date_info = ""
    if from_date or to_date:
        date_info = f" [{from_date or 'any'} → {to_date or 'any'}]"
    field_info = f" (field: {field})" if field != "all" else ""
    db_label = "PubMed Central" if db == "pmc" else "PubMed"
    print(f"\n[1] Searching {db_label}: '{term}' (max {max_results}){date_info}{field_info}...")

    base_params = {
        "db": db,
        "term": term,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        base_params["api_key"] = api_key

    # Quick count without date filter to inform the user
    count_params = {**base_params, "retmax": 0}
    if from_date or to_date:
        count_params["datetype"] = "pdat"
        if from_date:
            count_params["mindate"] = from_date
        if to_date:
            count_params["maxdate"] = to_date
    r = ncbi_get(f"{EUTILS}/esearch.fcgi", count_params)
    total = int(json.loads(r.text, strict=False)["esearchresult"]["count"])
    to_fetch = min(max_results, total)
    id_label = "PMCIDs" if db == "pmc" else "PMIDs"
    print(f"  Found {total} total results; will retrieve up to {to_fetch} {id_label}.")

    # PubMed hard-caps a single esearch at 9,999 records.
    # Workaround: sweep year-by-year and merge (deduplicating).
    from_year = int(from_date[:4]) if from_date else 1966
    to_year   = int(to_date[:4])   if to_date   else datetime.now().year

    seen: set[str] = set()
    pmids: list[str] = []
    for year in range(to_year, from_year - 1, -1):   # newest first
        if len(pmids) >= to_fetch:
            break
        want = to_fetch - len(pmids)
        batch = _esearch_year(base_params, year, want)
        new = [p for p in batch if p not in seen]
        seen.update(new)
        pmids.extend(new)
        print(f"  {year}: +{len(new)} PMIDs  (total {len(pmids)} / {to_fetch})", end="\r")
        time.sleep(0.11)

    print(f"  Retrieved {len(pmids)} {id_label}.{' ' * 40}")
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
    print(f"  Filtered to {len(reviews)} reviews / systematic reviews / meta-analyses.")
    return reviews


# ── Step 3: Citation counts via iCite ─────────────────────────────────────────

def add_citation_counts(articles: list[dict]) -> None:
    print(f"\n[3] Fetching citation counts from iCite for {len(articles)} articles...")
    pmids = [a["pmid"] for a in articles if a["pmid"]]
    counts: dict[str, int] = {}

    for batch in chunked(pmids, 100):
        r = requests.get(ICITE, params={"pmids": ",".join(batch)}, timeout=30)
        r.raise_for_status()
        for pub in json.loads(r.text, strict=False).get("data", []):
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

def load_pmcids_from_file(path: str) -> list[str]:
    """Read a plain-text file of PMCIDs (one per line or comma/space separated).
    Strips any leading 'PMC' prefix and blank lines."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    tokens = [t.strip() for t in raw.replace(",", " ").split()]
    pmcids = []
    for t in tokens:
        t = t.upper().removeprefix("PMC")
        if t.isdigit():
            pmcids.append(t)
    return pmcids


def pmcids_to_pmids(pmcids: list[str], email: str, api_key: str) -> list[str]:
    """Convert a list of numeric PMCIDs to PMIDs via NCBI ID converter."""
    print(f"\n[1] Converting {len(pmcids)} PMCIDs to PMIDs...")
    pmids: list[str] = []
    for batch in chunked(pmcids, 50):
        ids = ",".join(f"PMC{p}" for p in batch)
        params = {"ids": ids, "format": "json", "email": email}
        if api_key:
            params["tool"] = "pubmed_downloader"
        r = requests.get(IDCONV, params=params, timeout=30)
        if not r.ok:
            print(f"  Warning: ID converter returned {r.status_code} for a batch, skipping.")
            continue
        for rec in json.loads(r.text, strict=False).get("records", []):
            pmid = rec.get("pmid", "")
            if pmid:
                pmids.append(pmid)
        time.sleep(0.3)
    print(f"  Resolved {len(pmids)} / {len(pmcids)} PMIDs.")
    return pmids


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
        data = json.loads(r.text, strict=False)
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

    total = len(articles)
    no_pmc = 0
    not_oa = 0
    non_comm = 0
    failed = 0
    already = 0
    downloaded = 0

    for i, a in enumerate(articles):
        print(f"  Checking {i+1}/{total}...", end="\r")

        if not a["pmc_id"]:
            no_pmc += 1
            continue

        pdf_url, license_str = get_oa_info(a["pmc_id"])
        a["oa_license"] = license_str

        if not pdf_url:
            not_oa += 1
            time.sleep(0.2)
            continue

        if oa_comm_only and not is_commercial_license(license_str):
            non_comm += 1
            time.sleep(0.2)
            continue

        a["oa_pdf_url"] = pdf_url
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in a["title"])[:60]
        filename = output_dir / f"PMC{a['pmc_id']}_{safe_title}.pdf"

        if filename.exists():
            already += 1
            a["downloaded"] = True
            continue

        try:
            r = requests.get(pdf_url, timeout=60, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            title_short = a["title"][:80] + ("…" if len(a["title"]) > 80 else "")
            print(f"  [{downloaded + already + 1}] PMID {a['pmid']} — {title_short}")
            a["downloaded"] = True
            downloaded += 1
        except Exception as e:
            failed += 1
            print(f"  PMID {a['pmid']} — download failed: {e}")

        time.sleep(0.5)

    print(f"""
  ── Download summary ──────────────────────────
  Checked      : {total}
  No PMC ID    : {no_pmc}
  Not OA       : {not_oa}{"" if not oa_comm_only else f"  Non-commercial: {non_comm}"}
  Failed       : {failed}
  Already had  : {already}
  Downloaded   : {downloaded}
  ──────────────────────────────────────────────""")


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
    parser.add_argument("--query",         default="",     help='Search term, e.g. "tuberculosis drug resistance"')
    parser.add_argument("--pmcid-file",    default="",     help="Plain-text file of PMCIDs to use instead of searching (one per line or comma-separated)")
    parser.add_argument("--max-results",   type=int, default=500, help="Max PubMed records to retrieve when searching (default: 500)")
    parser.add_argument("--min-citations", type=int, default=10,  help="Minimum citation count to include (default: 10)")
    parser.add_argument("--from-date",     default="",     help="Earliest publication date, DD-MM-YYYY (optional)")
    parser.add_argument("--to-date",       default="",     help="Latest publication date, DD-MM-YYYY (optional)")
    parser.add_argument("--output-dir",    default="./reviews",   help="Folder to save PDFs and summary (default: ./reviews)")
    parser.add_argument("--email",         required=True,  help="Your email (required by NCBI)")
    parser.add_argument("--api-key",       default="2e107b339d3ce35ba53f430c0863f9245808", help="NCBI API key (optional; raises rate limit from 3 to 10 req/s)")
    parser.add_argument("--db",            default="pubmed", choices=["pubmed", "pmc"],
                                           help="NCBI database to search: pubmed (default) or pmc (PubMed Central)")
    parser.add_argument("--field",         default="all", choices=["all", "title", "tiab"],
                                           help="Restrict query to: all fields (default), title only, or title+abstract")
    parser.add_argument("--reviews-only",  action="store_true",   help="Keep only Review, Systematic Review, and Meta-Analysis article types")
    parser.add_argument("--no-download",   action="store_true",   help="Skip PDF download; only produce the summary TSV")
    parser.add_argument("--oa-comm-only",  action="store_true",   help="Only download PDFs with a commercial-friendly OA license (CC BY, CC BY-SA, CC BY-ND)")
    args = parser.parse_args()

    if not args.query and not args.pmcid_file:
        parser.error("Provide either --query or --pmcid-file.")

    output_dir = Path(args.output_dir)

    # 1. Search or load PMCIDs
    if args.pmcid_file:
        pmcids = load_pmcids_from_file(args.pmcid_file)
        if not pmcids:
            print("No valid PMCIDs found in file. Exiting.")
            sys.exit(0)
        pmids = pmcids_to_pmids(pmcids, args.email, args.api_key)
    else:
        from_date = parse_date(args.from_date, "from-date") if args.from_date else ""
        to_date   = parse_date(args.to_date,   "to-date")   if args.to_date   else ""
        ids = search_pubmed(args.query, args.max_results, args.email, args.api_key,
                            from_date=from_date, to_date=to_date, field=args.field,
                            db=args.db)
        if args.db == "pmc":
            # esearch on pmc returns PMCIDs; convert to PMIDs for the rest of the pipeline
            pmids = pmcids_to_pmids(ids, args.email, args.api_key)
        else:
            pmids = ids

    if not pmids:
        print("No results. Exiting.")
        sys.exit(0)

    # 2. Fetch details + filter reviews
    articles = fetch_article_details(pmids, args.email, args.api_key)
    if args.reviews_only:
        print("\n[2b] Filtering for reviews / systematic reviews / meta-analyses...")
        articles = filter_reviews(articles)
        if not articles:
            print("No reviews found. Try a broader query or remove --reviews-only.")
            sys.exit(0)

    # 3. Citation counts + filter
    add_citation_counts(articles)
    filtered = filter_by_citations(articles, args.min_citations)
    if not filtered:
        print(f"No articles with ≥ {args.min_citations} citations. Try lowering --min-citations.")
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

    print(f"\nDone. {len(filtered)} articles meet your criteria.")
    if not args.no_download:
        downloaded = sum(1 for a in filtered if a["downloaded"])
        print(f"  {downloaded} PDFs downloaded (rest are not open access).")


if __name__ == "__main__":
    main()
