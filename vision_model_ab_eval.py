#!/usr/bin/env python3
"""
Evaluate two Ollama vision models on scientific figures from PMC packages.

The evaluator:
- samples a reproducible set of PMC papers from an extracted review corpus
- runs the same figure-interpretation prompt against both models
- writes paired raw outputs plus lightweight aggregate metrics

This is intended for model-selection work around FIBLES figure retrieval.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

import pmc_indexer

PROMPT = (
    "Describe this scientific figure: its type (graph, microscopy image, diagram, etc.), "
    "key data or findings shown, axes and labels if present, and the main conclusion."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with",
}

AXIS_HINTS = ("axis", "axes", "x-axis", "y-axis", "x axis", "y axis", "label")
CONCLUSION_HINTS = ("conclusion", "suggest", "indicate", "show", "demonstrate", "support", "reveal")
FIGURE_TYPE_HINTS = (
    "graph", "plot", "chart", "microscopy", "image", "diagram", "schematic", "western blot",
    "heatmap", "histology", "immunofluorescence", "survival", "bar plot", "scatter", "line plot",
)
PROGRESS_BAR_WIDTH = 28


@dataclass
class FigureCase:
    paper_id: str
    xml_path: str
    figure_id: str
    label: str
    caption: str
    image_path: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def content_words(text: str) -> set[str]:
    return {tok for tok in tokenize(text) if len(tok) > 2 and tok not in STOPWORDS}


def prompt_score(response: str) -> dict[str, float | int]:
    lowered = response.lower()
    tokens = tokenize(response)
    token_set = set(tokens)
    return {
        "word_count": len(tokens),
        "mentions_axis_or_label": int(any(hint in lowered for hint in AXIS_HINTS)),
        "mentions_conclusion": int(any(hint in lowered for hint in CONCLUSION_HINTS)),
        "mentions_figure_type": int(any(hint in lowered for hint in FIGURE_TYPE_HINTS)),
        "unique_token_count": len(token_set),
    }


def caption_overlap(caption: str, response: str) -> dict[str, float | int]:
    caption_terms = content_words(caption)
    if not caption_terms:
        return {"caption_term_count": 0, "caption_recall": 0.0}
    response_terms = content_words(response)
    overlap = caption_terms & response_terms
    return {
        "caption_term_count": len(caption_terms),
        "caption_recall": round(len(overlap) / len(caption_terms), 4),
    }


def model_response(image_path: str, model: str, ollama_url: str, timeout_s: int) -> str:
    with open(image_path, "rb") as handle:
        image_b64 = pmc_indexer.base64.b64encode(handle.read()).decode()

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": PROMPT, "images": [image_b64], "stream": False},
        timeout=(5, timeout_s),
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def fetch_available_models(ollama_url: str) -> set[str]:
    response = requests.get(f"{ollama_url}/api/tags", timeout=(5, 30))
    response.raise_for_status()
    models = response.json().get("models", [])
    names = set()
    for model in models:
        for key in ("name", "model"):
            value = model.get(key)
            if value:
                names.add(value)
    return names


def ensure_model_available(model: str, ollama_url: str) -> None:
    available_models = fetch_available_models(ollama_url)
    if model in available_models:
        return

    print(f"[ollama] pulling missing model: {model}")
    response = requests.post(
        f"{ollama_url}/api/pull",
        json={"name": model, "stream": False},
        timeout=(5, 1800),
    )
    response.raise_for_status()

    available_models = fetch_available_models(ollama_url)
    if model not in available_models:
        raise RuntimeError(f"Model pull completed but '{model}' is still unavailable at {ollama_url}.")


def ensure_models_available(models: list[str], ollama_url: str) -> None:
    for model in models:
        ensure_model_available(model, ollama_url)


def store_paper_with_cached_vision(
    con: sqlite3.Connection,
    paper: pmc_indexer.PaperData,
    embed_model: str,
    ollama_url: str,
    figure_interpretations: dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with con:
        old_chunk_ids = [
            row[0]
            for row in con.execute("SELECT id FROM chunks WHERE paper_id=?", (paper.pmc_id,)).fetchall()
        ]
        if old_chunk_ids:
            con.executemany("DELETE FROM xrefs WHERE chunk_id=?", [(chunk_id,) for chunk_id in old_chunk_ids])
        con.execute("DELETE FROM chunks WHERE paper_id=?", (paper.pmc_id,))
        con.execute("DELETE FROM figures WHERE paper_id=?", (paper.pmc_id,))
        con.execute("DELETE FROM tables WHERE paper_id=?", (paper.pmc_id,))
        con.execute("DELETE FROM papers WHERE id=?", (paper.pmc_id,))

        con.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?)",
            (paper.pmc_id, paper.title, paper.authors, paper.year, paper.xml_path, int(time.time())),
        )

        for fig in paper.figures:
            row_id = str(pmc_indexer.uuid.uuid4())
            interp = figure_interpretations.get(fig.fig_id, "")
            con.execute(
                "INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?,?,?)",
                (row_id, paper.pmc_id, fig.fig_id, fig.label, fig.caption, fig.image_path, interp),
            )

            for text, ctype in ((fig.caption, "figure_legend"), (interp, "figure_interpretation")):
                for ct in pmc_indexer.chunk_words(text, chunk_size, chunk_overlap):
                    cid = str(pmc_indexer.uuid.uuid4())
                    con.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (
                            cid,
                            paper.pmc_id,
                            ctype,
                            fig.label,
                            fig.fig_id,
                            ct,
                            pmc_indexer.pack_emb(pmc_indexer.embed(ct, embed_model, ollama_url)),
                        ),
                    )

        for tbl in paper.tables:
            row_id = str(pmc_indexer.uuid.uuid4())
            con.execute(
                "INSERT OR REPLACE INTO tables VALUES (?,?,?,?,?,?)",
                (row_id, paper.pmc_id, tbl.table_id, tbl.label, tbl.caption, tbl.content_markdown),
            )

            for text, ctype in ((tbl.caption, "table_caption"), (tbl.content_markdown, "table_content")):
                for ct in pmc_indexer.chunk_words(text, chunk_size, chunk_overlap):
                    cid = str(pmc_indexer.uuid.uuid4())
                    con.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (
                            cid,
                            paper.pmc_id,
                            ctype,
                            tbl.label,
                            tbl.table_id,
                            ct,
                            pmc_indexer.pack_emb(pmc_indexer.embed(ct, embed_model, ollama_url)),
                        ),
                    )

        for sec_title, para in pmc_indexer.flatten_sections(paper.sections):
            if not para.text:
                continue
            chunk_ids = []
            for ct in pmc_indexer.chunk_words(para.text, chunk_size, chunk_overlap):
                cid = str(pmc_indexer.uuid.uuid4())
                chunk_ids.append(cid)
                con.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (
                        cid,
                        paper.pmc_id,
                        "section",
                        sec_title,
                        "",
                        ct,
                        pmc_indexer.pack_emb(pmc_indexer.embed(ct, embed_model, ollama_url)),
                    ),
                )
            for cid in chunk_ids:
                for xref in para.xrefs:
                    con.execute(
                        "INSERT OR IGNORE INTO xrefs VALUES (?,?,?)",
                        (cid, xref.ref_type, xref.rid),
                    )


def discover_cases(input_dir: Path) -> list[FigureCase]:
    cases: list[FigureCase] = []
    for paper_dir in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("PMC")):
        xmls = sorted(list(paper_dir.glob("*.nxml")) + list(paper_dir.glob("*.xml")))
        if not xmls:
            continue
        xml_path = xmls[0]
        try:
            paper = pmc_indexer.parse_jats(xml_path, paper_dir)
        except Exception as exc:
            print(f"[skip] {paper_dir.name}: failed to parse {xml_path.name}: {exc}", file=sys.stderr)
            continue

        for fig in paper.figures:
            if not fig.image_path:
                continue
            cases.append(
                FigureCase(
                    paper_id=paper.pmc_id,
                    xml_path=str(xml_path),
                    figure_id=fig.fig_id,
                    label=fig.label,
                    caption=fig.caption,
                    image_path=fig.image_path,
                )
            )
    return cases


def sample_cases(cases: list[FigureCase], paper_sample_size: int, seed: int) -> tuple[list[str], list[FigureCase]]:
    papers = sorted({case.paper_id for case in cases})
    if len(papers) < paper_sample_size:
        raise ValueError(f"Requested {paper_sample_size} papers but found only {len(papers)} with figures.")

    rng = random.Random(seed)
    sampled_papers = sorted(rng.sample(papers, paper_sample_size))
    sampled_cases = [case for case in cases if case.paper_id in set(sampled_papers)]
    return sampled_papers, sampled_cases


def aggregate_results(records: list[dict], model_name: str) -> dict[str, float | int]:
    model_records = [
        record["results"][model_name]
        for record in records
        if model_name in record.get("results", {})
    ]
    nonempty = [record for record in model_records if record["response"]]
    if not records:
        return {
            "cases": 0,
            "nonempty_rate": 0.0,
            "avg_word_count": 0.0,
            "avg_unique_token_count": 0.0,
            "axis_or_label_rate": 0.0,
            "conclusion_rate": 0.0,
            "figure_type_rate": 0.0,
            "avg_caption_recall": 0.0,
            "avg_latency_s": 0.0,
        }

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    return {
        "cases": len(model_records),
        "nonempty_rate": round(len(nonempty) / len(model_records), 4) if model_records else 0.0,
        "avg_word_count": mean([record["metrics"]["word_count"] for record in model_records]),
        "avg_unique_token_count": mean([record["metrics"]["unique_token_count"] for record in model_records]),
        "axis_or_label_rate": mean([record["metrics"]["mentions_axis_or_label"] for record in model_records]),
        "conclusion_rate": mean([record["metrics"]["mentions_conclusion"] for record in model_records]),
        "figure_type_rate": mean([record["metrics"]["mentions_figure_type"] for record in model_records]),
        "avg_caption_recall": mean([record["metrics"]["caption_recall"] for record in model_records]),
        "avg_latency_s": mean([record["latency_s"] for record in model_records]),
    }


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def make_case_key(case: FigureCase | dict) -> tuple[str, str]:
    return (str(case["paper_id"]) if isinstance(case, dict) else case.paper_id, str(case["figure_id"]) if isinstance(case, dict) else case.figure_id)


def has_saved_model_result(record: dict, model_name: str) -> bool:
    result = record.get("results", {}).get(model_name)
    return isinstance(result, dict) and "response" in result and "metrics" in result


def progress_bar(current: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0"
    clamped = min(max(current, 0), total)
    filled = int(width * clamped / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = (100.0 * clamped) / total
    return f"[{bar}] {clamped}/{total} ({pct:5.1f}%)"


class StatusPanel:
    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty()
        self.line_count = 0

    def render(
        self,
        paper_idx: int,
        total_papers: int,
        figure_idx: int,
        total_figures: int,
        paper_figure_idx: int,
        total_paper_figures: int,
        paper_id: str,
        figure_label: str,
        started_at: float,
    ) -> None:
        elapsed_s = int(max(0, time.time() - started_at))
        hours, rem = divmod(elapsed_s, 3600)
        minutes, seconds = divmod(rem, 60)
        timer = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        lines = [
            f"[paper]   {progress_bar(paper_idx, total_papers)} {paper_id}",
            f"[figure]  {progress_bar(figure_idx, total_figures)} {paper_id} {figure_label}",
            f"[inpaper] {progress_bar(paper_figure_idx, total_paper_figures)} {paper_id}",
            f"[elapsed] {timer}",
        ]

        if self.enabled:
            if self.line_count:
                sys.stdout.write(f"\x1b[{self.line_count}F")
            for line in lines:
                sys.stdout.write("\x1b[2K")
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self.line_count = len(lines)
            return

        for line in lines:
            print(line)

    def finish(self) -> None:
        if self.enabled and self.line_count:
            sys.stdout.write("\n")
            sys.stdout.flush()


def write_markdown_report(
    output_path: Path,
    sampled_papers: list[str],
    records: list[dict],
    summary: dict,
    models: list[str],
) -> None:
    lines = [
        "# Vision Model Evaluation",
        "",
        f"- Models: {', '.join(f'`{model}`' for model in models)}",
        f"- Sampled papers: {len(sampled_papers)}",
        f"- Completed papers: {summary['completed_papers']} / {len(sampled_papers)}",
        f"- Completed figure cases: {summary['completed_figure_cases']} / {summary['figure_case_count']}",
        f"- Seed: `{summary['seed']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | " + " | ".join(models) + " |",
        "| --- | " + " | ".join(["---:"] * len(models)) + " |",
    ]

    keys = [
        "nonempty_rate",
        "avg_word_count",
        "avg_unique_token_count",
        "axis_or_label_rate",
        "conclusion_rate",
        "figure_type_rate",
        "avg_caption_recall",
        "avg_latency_s",
    ]
    for key in keys:
        values = [str(summary["aggregate"][model][key]) for model in models]
        lines.append(f"| `{key}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Sampled Papers",
            "",
            ", ".join(sampled_papers),
            "",
            "## Per-Figure Outputs",
            "",
        ]
    )

    for idx, record in enumerate(records, start=1):
        case = record["case"]
        lines.extend(
            [
                f"### Case {idx}: {case['paper_id']} {case['label'] or case['figure_id']}",
                "",
                f"- Figure ID: `{case['figure_id']}`",
                f"- Image: `{case['image_path']}`",
                "",
                "**Caption**",
                "",
                case["caption"] or "_No caption_",
                "",
            ]
        )
        for model in models:
            result = record.get("results", {}).get(model, {})
            lines.extend(
                [
                    f"**Model: `{model}`**",
                    "",
                    result.get("response", "") or "_Empty response_",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    input_dir: Path,
    sampled_papers: list[str],
    total_figure_cases: int,
    records: list[dict],
    seed: int,
    completed_papers: int,
    models: list[str],
) -> dict:
    completed_cases = sum(
        1 for record in records
        if all(model in record.get("results", {}) for model in models)
    )
    return {
        "input_dir": str(input_dir),
        "prompt": PROMPT,
        "seed": seed,
        "sampled_papers": sampled_papers,
        "completed_papers": completed_papers,
        "remaining_papers": max(0, len(sampled_papers) - completed_papers),
        "completed_figure_cases": completed_cases,
        "figure_case_count": total_figure_cases,
        "models": models,
        "aggregate": {model: aggregate_results(records, model) for model in models},
    }


def write_reports(
    output_dir: Path,
    stem: str,
    sampled_papers: list[str],
    records: list[dict],
    summary: dict,
    models: list[str],
) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_payload = {
        "summary": summary,
        "records": records,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_markdown_report(md_path, sampled_papers, records, summary, models)
    return json_path, md_path


def load_existing_results(json_path: Path, sampled_papers: list[str]) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not json_path.exists():
        return [], {}

    data = json.loads(json_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    existing_sampled_papers = summary.get("sampled_papers")
    if existing_sampled_papers and existing_sampled_papers != sampled_papers:
        raise ValueError("Existing results file uses a different sampled paper set.")

    records = data.get("records", [])
    record_map = {
        make_case_key(record["case"]): record
        for record in records
    }
    return records, record_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="/home/k/Claude_Projects/Fibles/miscellaneous/reviews_010426_min2_2y",
        help="Directory containing extracted PMC paper folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma3:4b", "moondream2", "qwen2.5vl:7b", "llava-phi3", "llava-llama3:latest"],
        help="Vision models to run sequentially for each figure.",
    )
    parser.add_argument(
        "--paper-sample-size",
        type=int,
        default=30,
        help="Number of papers to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260401,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://172.31.80.1:11434",
        help="Base URL for the Ollama server.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=180,
        help="Per-request read timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/k/Claude_Projects/Fibles/miscellaneous/vision_model_eval",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text:latest",
        help="Ollama embedding model used when checkpointing per-model SQLite indexes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Chunk size used when checkpointing per-model SQLite indexes.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap used when checkpointing per-model SQLite indexes.",
    )
    parser.add_argument(
        "--write-index",
        action="store_true",
        help="Also checkpoint model-specific SQLite indexes after each paper completes.",
    )
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()
    models = args.models
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    required_models = list(models)
    if args.write_index:
        required_models.append(args.embed_model)
    print(f"[info] checking Ollama models at {args.ollama_url}")
    ensure_models_available(required_models, args.ollama_url)

    print(f"[info] discovering figures under {input_dir}")
    all_cases = discover_cases(input_dir)
    sampled_papers, sampled_cases = sample_cases(all_cases, args.paper_sample_size, args.seed)
    print(f"[info] sampled {len(sampled_papers)} papers -> {len(sampled_cases)} figure cases")

    cases_by_paper: dict[str, list[FigureCase]] = {}
    for case in sampled_cases:
        cases_by_paper.setdefault(case.paper_id, []).append(case)

    status_panel = StatusPanel()
    total_papers = len(sampled_papers)
    total_figures = len(sampled_cases)
    stem = (
        f"vision_eval_{'__'.join(sanitize_model_name(model) for model in models)}"
        f"_papers{args.paper_sample_size}_seed{args.seed}"
    )
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    records, record_map = load_existing_results(json_path, sampled_papers)
    completed_case_keys = {
        key for key, record in record_map.items()
        if all(has_saved_model_result(record, model) for model in models)
    }
    figure_idx = len(completed_case_keys)
    db_paths = {
        model: output_dir / f"{stem}_{sanitize_model_name(model)}.db"
        for model in models
    }
    connections = {
        model: pmc_indexer.open_db(str(path))
        for model, path in db_paths.items()
    } if args.write_index else {}
    indexed_ids = {
        model: {row[0] for row in con.execute("SELECT id FROM papers").fetchall()}
        for model, con in connections.items()
    }
    for paper_idx, paper_id in enumerate(sampled_papers, start=1):
        paper_cases = cases_by_paper.get(paper_id, [])
        paper_figure_total = len(paper_cases)
        paper = None
        if paper_cases:
            paper_dir = Path(paper_cases[0].xml_path).parent
            paper = pmc_indexer.parse_jats(Path(paper_cases[0].xml_path), paper_dir)
        for paper_figure_idx, case in enumerate(paper_cases, start=1):
            case_key = make_case_key(case)
            record = record_map.get(case_key)
            if record is None:
                record = {"case": asdict(case), "results": {}}
                record_map[case_key] = record
                records.append(record)
            elif "results" not in record:
                record["results"] = {}

            if not all(has_saved_model_result(record, model) for model in models):
                figure_idx += 1
            status_panel.render(
                paper_idx=paper_idx,
                total_papers=total_papers,
                figure_idx=figure_idx,
                total_figures=total_figures,
                paper_figure_idx=paper_figure_idx,
                total_paper_figures=paper_figure_total,
                paper_id=case.paper_id,
                figure_label=case.label or case.figure_id,
                started_at=started_at,
            )
            for model_name in models:
                if has_saved_model_result(record, model_name):
                    continue
                started = time.time()
                try:
                    response = model_response(case.image_path, model_name, args.ollama_url, args.timeout_s)
                    error = ""
                except Exception as exc:
                    response = ""
                    error = str(exc)
                latency_s = round(time.time() - started, 3)
                metrics = prompt_score(response)
                metrics.update(caption_overlap(case.caption, response))
                record["results"][model_name] = {
                    "model": model_name,
                    "latency_s": latency_s,
                    "error": error,
                    "response": response,
                    "metrics": metrics,
                }
        if args.write_index and paper is not None:
            paper_records = {
                make_case_key(record["case"]): record
                for record in records
                if record["case"]["paper_id"] == paper_id
            }
            for model_name in models:
                if paper_id in indexed_ids.get(model_name, set()):
                    continue
                if not all(
                    has_saved_model_result(paper_records.get(make_case_key(case), {}), model_name)
                    for case in paper_cases
                ):
                    continue
                figure_interpretations = {
                    case.figure_id: paper_records[make_case_key(case)]["results"][model_name]["response"]
                    for case in paper_cases
                }
                store_paper_with_cached_vision(
                    con=connections[model_name],
                    paper=paper,
                    embed_model=args.embed_model,
                    ollama_url=args.ollama_url,
                    figure_interpretations=figure_interpretations,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                indexed_ids[model_name].add(paper_id)
        summary = build_summary(
            input_dir=input_dir,
            sampled_papers=sampled_papers,
            total_figure_cases=total_figures,
            records=records,
            seed=args.seed,
            completed_papers=paper_idx,
            models=models,
        )
        json_path, md_path = write_reports(
            output_dir=output_dir,
            stem=stem,
            sampled_papers=sampled_papers,
            records=records,
            summary=summary,
            models=models,
        )
    status_panel.finish()
    for con in connections.values():
        con.close()

    summary = build_summary(
        input_dir=input_dir,
        sampled_papers=sampled_papers,
        total_figure_cases=total_figures,
        records=records,
        seed=args.seed,
        completed_papers=total_papers,
        models=models,
    )
    json_path, md_path = write_reports(
        output_dir=output_dir,
        stem=stem,
        sampled_papers=sampled_papers,
        records=records,
        summary=summary,
        models=models,
    )

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")
    if args.write_index:
        for path in db_paths.values():
            print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
