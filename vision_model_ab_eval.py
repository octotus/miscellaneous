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


def aggregate_results(records: list[dict], model_key: str) -> dict[str, float | int]:
    nonempty = [record for record in records if record[model_key]["response"]]
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
        "cases": len(records),
        "nonempty_rate": round(len(nonempty) / len(records), 4),
        "avg_word_count": mean([record[model_key]["metrics"]["word_count"] for record in records]),
        "avg_unique_token_count": mean([record[model_key]["metrics"]["unique_token_count"] for record in records]),
        "axis_or_label_rate": mean([record[model_key]["metrics"]["mentions_axis_or_label"] for record in records]),
        "conclusion_rate": mean([record[model_key]["metrics"]["mentions_conclusion"] for record in records]),
        "figure_type_rate": mean([record[model_key]["metrics"]["mentions_figure_type"] for record in records]),
        "avg_caption_recall": mean([record[model_key]["metrics"]["caption_recall"] for record in records]),
        "avg_latency_s": mean([record[model_key]["latency_s"] for record in records]),
    }


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


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
    model_a: str,
    model_b: str,
) -> None:
    lines = [
        "# Vision Model A/B Evaluation",
        "",
        f"- Model A: `{model_a}`",
        f"- Model B: `{model_b}`",
        f"- Sampled papers: {len(sampled_papers)}",
        f"- Figure cases: {len(records)}",
        f"- Seed: `{summary['seed']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Model A | Model B |",
        "| --- | ---: | ---: |",
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
        lines.append(f"| `{key}` | {summary['aggregate']['model_a'][key]} | {summary['aggregate']['model_b'][key]} |")

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
                f"**Model A: `{model_a}`**",
                "",
                record["model_a"]["response"] or "_Empty response_",
                "",
                f"**Model B: `{model_b}`**",
                "",
                record["model_b"]["response"] or "_Empty response_",
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="/home/k/Claude_Projects/Fibles/miscellaneous/reviews_010426_min2_2y",
        help="Directory containing extracted PMC paper folders.",
    )
    parser.add_argument(
        "--model-a",
        default="gemma3:4b",
        help="First Ollama vision model.",
    )
    parser.add_argument(
        "--model-b",
        default="moondream2",
        help="Second Ollama vision model.",
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
        default="/home/k/Claude_Projects/Fibles/miscellaneous/vision_eval_outputs",
        help="Directory for JSON and Markdown reports.",
    )
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    print(f"[info] discovering figures under {input_dir}")
    all_cases = discover_cases(input_dir)
    sampled_papers, sampled_cases = sample_cases(all_cases, args.paper_sample_size, args.seed)
    print(f"[info] sampled {len(sampled_papers)} papers -> {len(sampled_cases)} figure cases")

    cases_by_paper: dict[str, list[FigureCase]] = {}
    for case in sampled_cases:
        cases_by_paper.setdefault(case.paper_id, []).append(case)

    status_panel = StatusPanel()
    records: list[dict] = []
    figure_idx = 0
    total_papers = len(sampled_papers)
    total_figures = len(sampled_cases)
    for paper_idx, paper_id in enumerate(sampled_papers, start=1):
        paper_cases = cases_by_paper.get(paper_id, [])
        paper_figure_total = len(paper_cases)
        for paper_figure_idx, case in enumerate(paper_cases, start=1):
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
            row = {"case": asdict(case)}
            for model_key, model_name in (("model_a", args.model_a), ("model_b", args.model_b)):
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
                row[model_key] = {
                    "model": model_name,
                    "latency_s": latency_s,
                    "error": error,
                    "response": response,
                    "metrics": metrics,
                }
            records.append(row)
    status_panel.finish()

    summary = {
        "input_dir": str(input_dir),
        "prompt": PROMPT,
        "seed": args.seed,
        "sampled_papers": sampled_papers,
        "figure_case_count": len(sampled_cases),
        "aggregate": {
            "model_a": aggregate_results(records, "model_a"),
            "model_b": aggregate_results(records, "model_b"),
        },
    }

    stem = (
        f"vision_ab_eval_{sanitize_model_name(args.model_a)}"
        f"_vs_{sanitize_model_name(args.model_b)}"
        f"_papers{args.paper_sample_size}_seed{args.seed}"
    )
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_payload = {
        "summary": summary,
        "records": records,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_markdown_report(md_path, sampled_papers, records, summary, args.model_a, args.model_b)

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
