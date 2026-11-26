"""Summarize the benchmark files in ``results/`` into a single table.

Each run of ``python -m src.main`` writes one ``results/<model><optimization>_<tag>.txt`` file.
This script parses all of them, keeps the runs evaluated on the same number of test images
(64 by default) and prints a Markdown table comparing plaintext inference, the unoptimized
IPFE baseline and the four parallelization strategies.

Usage::

    uv run python scripts/summarize_results.py               # Markdown table to stdout
    uv run python scripts/summarize_results.py --out docs/results.md --csv docs/results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Display name for each optimization suffix written by ``src.main.save_metrics_to_txt``.
VARIANTS = {
    "plain": "Plaintext",
    "ipfe": "IPFE (sequential)",
    "ipfe_kernel_parallelization": "Kernel threads",
    "ipfe_kernel_patches_parallelization": "Kernel + patch threads",
    "ipfe_batch_parallelization": "Batched (Numba)",
    "ipfe_batch_kernels_parallelization": "Batched (Numba) + kernel threads",
}

TAG_RE = re.compile(r"k(?P<k>\d+)_conv(?P<conv>[\d_]+)_stride(?P<stride>\d+)_pad(?P<pad>\d+)")


@dataclass
class Run:
    variant: str
    tag: str
    params: int
    test_size: int
    top1: float
    eval_seconds: float
    encrypt_seconds: float | None


def parse_file(path: Path) -> Run | None:
    stem = path.stem
    variant = next((v for v in sorted(VARIANTS, key=len, reverse=True) if stem.startswith(v + "_")), None)
    if variant is None:
        return None
    fields = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    try:
        return Run(
            variant=variant,
            tag=fields["Tag"],
            params=int(fields["Total Parameters"].replace(",", "")),
            test_size=int(fields["Test set size"]),
            top1=float(fields["Test top1"]),
            eval_seconds=float(fields["Test eval_seconds"]),
            encrypt_seconds=float(fields["Test encrypt_seconds"]) if "Test encrypt_seconds" in fields else None,
        )
    except (KeyError, ValueError):
        return None


def load_runs(results_dir: Path, test_size: int) -> dict[str, dict[str, Run]]:
    """Return ``{tag: {variant: run}}`` for all runs with the given test-set size."""
    table: dict[str, dict[str, Run]] = {}
    for path in sorted(results_dir.glob("*.txt")):
        run = parse_file(path)
        if run is not None and run.test_size == test_size:
            table.setdefault(run.tag, {})[run.variant] = run
    return table


def describe(tag: str) -> str:
    m = TAG_RE.match(tag)
    if not m:
        return tag
    return f"{m['k']}×{m['k']} kernel, stride {m['stride']}, pad {m['pad']}, channels {m['conv'].replace('_', '-')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--out", type=Path, help="write the Markdown table to this file instead of stdout")
    parser.add_argument("--csv", type=Path, help="also write every parsed run to this CSV file")
    args = parser.parse_args()

    table = load_runs(args.results, args.test_size)
    order = list(VARIANTS)

    header = ["Model", "Params", "Plain acc.", "IPFE acc."] + [f"{VARIANTS[v]} (s)" for v in order]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["--:"] * (len(header) - 1)) + "|"]

    rows = []
    for tag in sorted(table):
        runs = table[tag]
        any_run = next(iter(runs.values()))
        ipfe_accs = {r.top1 for v, r in runs.items() if v != "plain"}
        ipfe_acc = f"{min(ipfe_accs):.1f}%" if ipfe_accs else "–"  # worst case over IPFE variants
        plain_acc = f"{runs['plain'].top1:.1f}%" if "plain" in runs else "–"
        times = [f"{runs[v].eval_seconds:.1f}" if v in runs else "–" for v in order]
        lines.append(f"| `{tag}` | {any_run.params:,} | {plain_acc} | {ipfe_acc} | " + " | ".join(times) + " |")
        rows.extend(runs.values())

    markdown = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"Wrote table for {len(table)} models to {args.out}")
    else:
        print(markdown, end="")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["tag", "description", "variant", "params", "test_size", "top1", "eval_seconds", "encrypt_seconds"]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.tag,
                        describe(r.tag),
                        r.variant,
                        r.params,
                        r.test_size,
                        r.top1,
                        r.eval_seconds,
                        r.encrypt_seconds,
                    ]
                )
        print(f"\nWrote {len(rows)} runs to {args.csv}")


if __name__ == "__main__":
    main()
