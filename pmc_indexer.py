#!/usr/bin/env python3
"""
PMC Indexer
-----------
Indexes PMC full-text packages (XML + figures + tables) extracted by
pubmed_review_downloader.py --format tgz into a searchable SQLite database.

Usage:
    python pmc_indexer.py index  --input-dir ./reviews [options]
    python pmc_indexer.py search --query "drug resistance" [options]

Requirements:
    pip install requests numpy
"""

import argparse
import base64
import json
import re
import sqlite3
import struct
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

# ── Constants ──────────────────────────────────────────────────────────────────

XLINK_NS   = "http://www.w3.org/1999/xlink"
IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".gif"}

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class XRef:
    ref_type: str   # "fig" or "table"
    rid: str

@dataclass
class Paragraph:
    text: str
    xrefs: list[XRef] = field(default_factory=list)

@dataclass
class Section:
    title: str
    level: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)

@dataclass
class Figure:
    fig_id: str
    label: str
    caption: str
    image_path: str   # absolute path, empty if not found

@dataclass
class Table:
    table_id: str
    label: str
    caption: str
    content_markdown: str

@dataclass
class PaperData:
    pmc_id: str
    title: str
    authors: str
    year: str
    xml_path: str
    sections: list[Section] = field(default_factory=list)
    figures: list[Figure]   = field(default_factory=list)
    tables: list[Table]     = field(default_factory=list)

# ── Database ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS papers (
  id TEXT PRIMARY KEY,
  title TEXT, authors TEXT, year TEXT, xml_path TEXT, indexed_at INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  chunk_type TEXT,
  section_title TEXT,
  element_id TEXT,
  text TEXT,
  embedding BLOB,
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE TABLE IF NOT EXISTS figures (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  fig_id TEXT,
  label TEXT,
  caption TEXT,
  image_path TEXT,
  interpretation TEXT,
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE TABLE IF NOT EXISTS tables (
  id TEXT PRIMARY KEY,
  paper_id TEXT,
  table_id TEXT,
  label TEXT,
  caption TEXT,
  content_markdown TEXT,
  FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE TABLE IF NOT EXISTS xrefs (
  chunk_id TEXT,
  ref_type TEXT,
  target_id TEXT,
  PRIMARY KEY (chunk_id, ref_type, target_id)
);
"""

def open_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.executescript(DDL)
    con.commit()
    return con

# ── JATS XML parsing ───────────────────────────────────────────────────────────

def _clean(elem) -> str:
    return " ".join("".join(elem.itertext()).split())

def _find_image(paper_dir: Path, href: str) -> str:
    stem = Path(href).stem
    for candidate in [paper_dir / href] + [paper_dir / f"{stem}{ext}" for ext in IMAGE_EXTS]:
        if candidate.exists():
            return str(candidate)
    for f in paper_dir.rglob(stem + "*"):
        if f.suffix.lower() in IMAGE_EXTS:
            return str(f)
    return ""

def _parse_paragraph(p_elem) -> Paragraph:
    text = _clean(p_elem)
    xrefs = [
        XRef(ref_type=x.get("ref-type", ""), rid=x.get("rid", ""))
        for x in p_elem.findall(".//xref")
        if x.get("ref-type") in ("fig", "table") and x.get("rid")
    ]
    return Paragraph(text=text, xrefs=xrefs)

def _parse_sec(sec_elem, level: int = 1) -> Section:
    title_elem = sec_elem.find("title")
    title = _clean(title_elem) if title_elem is not None else ""
    return Section(
        title=title,
        level=level,
        paragraphs=[_parse_paragraph(p) for p in sec_elem.findall("p")],
        subsections=[_parse_sec(s, level + 1) for s in sec_elem.findall("sec")],
    )

def _table_to_markdown(table_elem) -> str:
    rows = []
    header_done = False
    for tr in table_elem.findall(".//tr"):
        cells = [_clean(c) for c in tr.findall("th") + tr.findall("td")]
        if not cells:
            continue
        rows.append("| " + " | ".join(cells) + " |")
        if not header_done:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            header_done = True
    return "\n".join(rows)

def parse_jats(xml_path: Path, paper_dir: Path) -> PaperData:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    front = root.find(".//front")

    # Metadata
    title = ""
    if front is not None:
        t = front.find(".//article-title")
        if t is not None:
            title = _clean(t)

    authors = []
    if front is not None:
        for contrib in front.findall(".//contrib[@contrib-type='author']"):
            surname  = contrib.findtext(".//surname", "")
            given    = contrib.findtext(".//given-names", "")
            if surname:
                authors.append(f"{surname} {given[0]}." if given else surname)
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

    year = ""
    if front is not None:
        year = (front.findtext(".//pub-date/year") or
                front.findtext(".//pub-date[@pub-type='epub']/year") or "")

    # Body sections
    body = root.find(".//body")
    sections = [_parse_sec(s) for s in (body.findall("sec") if body is not None else [])]

    # Figures
    figures = []
    for fig in root.findall(".//fig"):
        fig_id  = fig.get("id", "")
        label   = _clean(fig.find("label")) if fig.find("label") is not None else ""
        cap_el  = fig.find(".//caption")
        caption = _clean(cap_el) if cap_el is not None else ""
        graphic = fig.find("graphic")
        href    = graphic.get(f"{{{XLINK_NS}}}href", "") if graphic is not None else ""
        image_path = _find_image(paper_dir, href) if href else ""
        figures.append(Figure(fig_id=fig_id, label=label, caption=caption, image_path=image_path))

    # Tables
    tables = []
    for tw in root.findall(".//table-wrap"):
        table_id = tw.get("id", "")
        label    = _clean(tw.find("label")) if tw.find("label") is not None else ""
        cap_el   = tw.find(".//caption")
        caption  = _clean(cap_el) if cap_el is not None else ""
        tbl_el   = tw.find(".//table")
        content  = _table_to_markdown(tbl_el) if tbl_el is not None else ""
        tables.append(Table(table_id=table_id, label=label, caption=caption, content_markdown=content))

    pmc_id = re.sub(r'^PMC', '', paper_dir.name, flags=re.IGNORECASE)
    return PaperData(pmc_id=pmc_id, title=title, authors=author_str, year=year,
                     xml_path=str(xml_path), sections=sections, figures=figures, tables=tables)

# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_words(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks

# ── Ollama ─────────────────────────────────────────────────────────────────────

def check_ollama(ollama_url: str) -> None:
    """Verify Ollama is reachable; exit with a clear message if not."""
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=(5, 10))
        r.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"\nCannot reach Ollama at {ollama_url}")
        print("  - Check the URL and port (default: 11434)")
        print("  - If Ollama is on another machine, ensure it is bound to 0.0.0.0:")
        print("    Windows: set OLLAMA_HOST=0.0.0.0 in system env vars, then restart Ollama")
        print("    Linux:   OLLAMA_HOST=0.0.0.0 ollama serve")
        print(f"  - Raw error: {e}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\nOllama at {ollama_url} did not respond within 5 seconds.")
        print("  - The host may be firewalled or the port blocked.")
        sys.exit(1)

def embed(text: str, model: str, ollama_url: str) -> list[float]:
    r = requests.post(f"{ollama_url}/api/embeddings",
                      json={"model": model, "prompt": text}, timeout=(5, 120))
    r.raise_for_status()
    return r.json()["embedding"]

def pack_emb(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)

def unpack_emb(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

def interpret_figure(image_path: str, model: str, ollama_url: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = ("Describe this scientific figure: its type (graph, microscopy image, diagram, etc.), "
              "key data or findings shown, axes and labels if present, and the main conclusion.")
    r = requests.post(f"{ollama_url}/api/generate",
                      json={"model": model, "prompt": prompt, "images": [b64], "stream": False},
                      timeout=(5, 180))
    r.raise_for_status()
    return r.json().get("response", "").strip()

# ── Flatten sections ───────────────────────────────────────────────────────────

def flatten_sections(sections: list[Section], parent: str = "") -> list[tuple[str, Paragraph]]:
    result = []
    for sec in sections:
        full = f"{parent} > {sec.title}".lstrip(" > ") if sec.title else parent
        for para in sec.paragraphs:
            result.append((full, para))
        result.extend(flatten_sections(sec.subsections, full))
    return result

# ── Store ──────────────────────────────────────────────────────────────────────

def store_paper(con: sqlite3.Connection, paper: PaperData,
                embed_model: str, ollama_url: str,
                vision_model: str, skip_vision: bool,
                chunk_size: int, chunk_overlap: int) -> None:
    with con:
        con.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?)",
                    (paper.pmc_id, paper.title, paper.authors, paper.year,
                     paper.xml_path, int(time.time())))

        # Figures
        for fig in paper.figures:
            row_id = str(uuid.uuid4())
            interp = ""
            if not skip_vision and fig.image_path:
                try:
                    interp = interpret_figure(fig.image_path, vision_model, ollama_url)
                except Exception as e:
                    print(f"\n    [vision] {fig.fig_id}: {e}")

            con.execute("INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?,?,?)",
                        (row_id, paper.pmc_id, fig.fig_id, fig.label,
                         fig.caption, fig.image_path, interp))

            for text, ctype in [(fig.caption, "figure_legend"), (interp, "figure_interpretation")]:
                for ct in chunk_words(text, chunk_size, chunk_overlap):
                    cid = str(uuid.uuid4())
                    con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                                (cid, paper.pmc_id, ctype, fig.label, fig.fig_id,
                                 ct, pack_emb(embed(ct, embed_model, ollama_url))))

        # Tables
        for tbl in paper.tables:
            row_id = str(uuid.uuid4())
            con.execute("INSERT OR REPLACE INTO tables VALUES (?,?,?,?,?,?)",
                        (row_id, paper.pmc_id, tbl.table_id, tbl.label,
                         tbl.caption, tbl.content_markdown))

            for text, ctype in [(tbl.caption, "table_caption"),
                                 (tbl.content_markdown, "table_content")]:
                for ct in chunk_words(text, chunk_size, chunk_overlap):
                    cid = str(uuid.uuid4())
                    con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                                (cid, paper.pmc_id, ctype, tbl.label, tbl.table_id,
                                 ct, pack_emb(embed(ct, embed_model, ollama_url))))

        # Section text + xrefs
        for sec_title, para in flatten_sections(paper.sections):
            if not para.text:
                continue
            for ct in chunk_words(para.text, chunk_size, chunk_overlap):
                cid = str(uuid.uuid4())
                con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                            (cid, paper.pmc_id, "section", sec_title, "",
                             ct, pack_emb(embed(ct, embed_model, ollama_url))))
                for xref in para.xrefs:
                    con.execute("INSERT OR IGNORE INTO xrefs VALUES (?,?,?)",
                                (cid, xref.ref_type, xref.rid))

# ── Search ─────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

def cmd_search(args) -> None:
    check_ollama(args.ollama_url)
    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    con = sqlite3.connect(args.db)
    q_emb = np.array(embed(args.query, args.embed_model, args.ollama_url), dtype=np.float32)

    rows = con.execute(
        "SELECT id, paper_id, chunk_type, section_title, element_id, text, embedding FROM chunks"
    ).fetchall()

    scored = []
    for cid, paper_id, ctype, sec_title, elem_id, text, blob in rows:
        if blob:
            scored.append((cosine(q_emb, unpack_emb(blob)),
                           cid, paper_id, ctype, sec_title, text))
    scored.sort(reverse=True)

    for rank, (score, cid, paper_id, ctype, sec_title, text) in enumerate(scored[:args.top_k], 1):
        loc = f"PMC{paper_id} | {sec_title or ctype}"
        print(f"\n[{rank}] {loc}   score={score:.3f}")
        print(f"    {text[:240]}{'…' if len(text) > 240 else ''}")

        for ref_type, target_id in con.execute(
            "SELECT ref_type, target_id FROM xrefs WHERE chunk_id=?", (cid,)
        ).fetchall():
            if ref_type == "fig":
                row = con.execute(
                    "SELECT label, caption, image_path, interpretation FROM figures "
                    "WHERE paper_id=? AND fig_id=?", (paper_id, target_id)
                ).fetchone()
                if row:
                    label, cap, img, interp = row
                    print(f"    -> {label}: {(interp or cap)[:160]}")
                    if img:
                        print(f"       Image: {img}")
            elif ref_type == "table":
                row = con.execute(
                    "SELECT label, caption, content_markdown FROM tables "
                    "WHERE paper_id=? AND table_id=?", (paper_id, target_id)
                ).fetchone()
                if row:
                    label, cap, md = row
                    print(f"    -> {label}: {(cap or md or '')[:160]}")

    con.close()

# ── Index ──────────────────────────────────────────────────────────────────────

def cmd_index(args) -> None:
    check_ollama(args.ollama_url)
    input_dir  = Path(args.input_dir)
    if args.db is None:
        args.db = str(input_dir.resolve().name) + ".db"
    # Find all PMC{id}/ dirs at any depth; deduplicate by pmc_id keeping the
    # one that actually contains an XML/NXML file (handles old nested extractions).
    def find_xml(d: Path):
        return [f for f in d.iterdir() if f.is_file() and f.suffix in ('.xml', '.nxml')]

    seen: dict[str, tuple[Path, list]] = {}
    for d in input_dir.rglob("PMC*"):
        if not (d.is_dir() and re.match(r'^PMC\d+$', d.name, re.IGNORECASE)):
            continue
        pmc_id = re.sub(r'^PMC', '', d.name, flags=re.IGNORECASE)
        xmls = find_xml(d)
        if pmc_id not in seen or (xmls and not seen[pmc_id][1]):
            seen[pmc_id] = (d, xmls)

    paper_entries = sorted(seen.items())   # [(pmc_id, (paper_dir, xml_files))]
    if not paper_entries:
        print(f"No PMC{{id}} subdirectories found under '{input_dir}'.")
        sys.exit(1)

    con = open_db(args.db)
    indexed_ids = {r[0] for r in con.execute("SELECT id FROM papers").fetchall()}

    total, done, skipped, failed = len(paper_entries), 0, 0, 0

    for i, (pmc_id, (paper_dir, xml_files)) in enumerate(paper_entries, 1):
        print(f"  [{i}/{total}] PMC{pmc_id}", end="  ", flush=True)

        if pmc_id in indexed_ids and not args.reindex:
            print("already indexed.")
            skipped += 1
            continue

        if not xml_files:
            print("no XML found, skipping.")
            failed += 1
            continue

        try:
            paper = parse_jats(xml_files[0], paper_dir)
            store_paper(con, paper,
                        embed_model=args.embed_model,
                        ollama_url=args.ollama_url,
                        vision_model=args.vision_model,
                        skip_vision=args.skip_vision,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap)
            done += 1
            print(f"OK  ({len(paper.sections)} sections, "
                  f"{len(paper.figures)} figures, {len(paper.tables)} tables)")
        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")

    con.close()
    print(f"\nDone.  Indexed: {done}   Skipped: {skipped}   Failed: {failed}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _ollama_args = dict(
        flags=["--ollama-url"],
        kwargs=dict(default="http://localhost:11434",
                    help="Ollama base URL (default: http://localhost:11434). "
                         "On WSL2 with Ollama on Windows, use http://<windows-ip>:11434"),
    )
    _shared = argparse.ArgumentParser(add_help=False)
    _shared.add_argument("--db",          default=None,               help="SQLite database path (default: <input-dir-name>.db for index, required for search)")
    _shared.add_argument("--ollama-url",  **_ollama_args["kwargs"])
    _shared.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")

    parser = argparse.ArgumentParser(description="Index and search PMC full-text packages.",
                                     parents=[_shared])
    sub = parser.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", parents=[_shared], help="Index a directory of extracted PMC packages")
    pi.add_argument("--input-dir",     required=True,         help="Folder containing PMC{id}/ subdirectories")
    pi.add_argument("--vision-model",  default="llava",       help="Ollama vision model for figure interpretation (default: llava)")
    pi.add_argument("--chunk-size",    type=int, default=400, help="Max words per chunk (default: 400)")
    pi.add_argument("--chunk-overlap", type=int, default=50,  help="Word overlap between chunks (default: 50)")
    pi.add_argument("--skip-vision",   action="store_true",   help="Skip figure interpretation")
    pi.add_argument("--reindex",       action="store_true",   help="Reprocess already-indexed papers")

    ps = sub.add_parser("search", parents=[_shared], help="Search the index")
    ps.add_argument("--query",  required=True,        help="Search query")
    ps.add_argument("--top-k",  type=int, default=10, help="Number of results (default: 10)")
    ps.set_defaults(db_required=True)

    args = parser.parse_args()
    if getattr(args, "db_required", False) and args.db is None:
        parser.error("--db is required for the search subcommand")
    if args.cmd == "index":
        cmd_index(args)
    else:
        cmd_search(args)


if __name__ == "__main__":
    main()
