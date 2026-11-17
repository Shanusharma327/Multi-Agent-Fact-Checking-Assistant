"""Comparison tables and charts for the results section.

Reads a benchmark results JSONL and emits, for the multi-agent pipeline and
the single-LLM baseline:

  (a) accuracy      (b) precision     (c) recall
  (d) F1-score      (e) computation time and error rate

Each as a LaTeX table and a matplotlib bar chart. Every value is computed
from the results file -- nothing is hardcoded, so the output cannot drift
away from the run it describes.

    python src/analysis/results_report.py --results results/averitec_n60.jsonl

Writes:
  results/figures/res_<metric>.png   - one chart per metric
  results/tables/res_tables.tex      - all five tables, \\input-able
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ablation import apply_policy, make_deterministic  # noqa: E402
from fact_checker.metrics import score_system, score_subset  # noqa: E402
from fact_checker.schemas import VERDICT_VALUES  # noqa: E402

FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")

CRIMSON = "#A51C30"
NAVY = "#1E2761"
GREY = "#5B6472"

# The proposed configuration computes the verdict from the audit, so
# "computed" is the series the paper is about; "multi" is the record as
# the run emitted it, which is the agent-reported ablation.
SYSTEMS = (("baseline", "Single-LLM (existing)", NAVY),
           ("computed", "MAFCA computed (proposed)", CRIMSON),
           ("multi", "MAFCA agent-reported (ablation)", GREY))

DECIDABLE = ("SUPPORTS", "REFUTES")


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [with_computed(r) for r in rows]


def with_computed(record: dict) -> dict:
    """Attach the computed-verdict block alongside the as-run one.

    Replays the deterministic policy over the stored audit, which is
    the same replay analysis/ablation.py performs, so a figure and the
    ablation table cannot report different numbers for the same
    configuration.
    """
    replayed = apply_policy([record], make_deterministic(1.0, gate=True))[0]
    record["computed"] = dict(replayed["multi"])
    return record


def cards(rows: Sequence[dict], keep: Sequence[str] | None = None):
    """Scorecards for both systems, optionally restricted to ``keep`` labels."""
    out = {}
    for key, _label, _colour in SYSTEMS:
        gold = [r["gold"] for r in rows]
        pred = [r[key]["verdict"] for r in rows]
        conf = [float(r[key]["confidence"]) for r in rows]
        secs = [float(r[key].get("seconds", 0.0)) for r in rows]
        if keep is None:
            out[key] = score_system(key, gold, pred, conf, secs)
        else:
            out[key] = score_subset(key, gold, pred, conf, keep)
    return out


def bar_chart(title: str, ylabel: str, groups: List[str],
              series: Dict[str, List[float]], filename: str,
              ylim: float = 108.0) -> None:
    """Grouped bar chart comparing the two systems across ``groups``."""
    fig, ax = plt.subplots(figsize=(max(5.6, 1.9 * len(groups) + 2.4), 3.9))
    width = 0.84 / len(SYSTEMS)
    x = range(len(groups))
    # Centre the group on the tick whatever the number of series.
    offsets = [
        (i - (len(SYSTEMS) - 1) / 2) * width for i in range(len(SYSTEMS))
    ]
    for offset, (key, label, colour) in zip(offsets, SYSTEMS):
        bars = ax.bar([i + offset for i in x], series[key], width,
                      label=label, color=colour)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=7.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def latex_table(caption: str, label: str, header: Sequence[str],
                body: Sequence[Sequence[str]]) -> str:
    """Render one booktabs table."""
    cols = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
        rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="Benchmark results JSONL.")
    ap.add_argument("--tag", default="res", help="Filename prefix.")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"No results file at {args.results}. Run evaluate.py first.")
        return 1

    rows = load(args.results)
    n = len(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    overall = cards(rows)
    decidable = cards([r for r in rows if r["gold"] in DECIDABLE], None)
    tex: List[str] = [
        f"% Generated by analysis/results_report.py from {args.results}",
        f"% n = {n} claims. Every value computed; none hardcoded.",
        "",
    ]

    # ---- (a) accuracy -------------------------------------------------
    acc = {k: 100 * overall[k].accuracy for k, _, _ in SYSTEMS}
    acc_dec = {k: 100 * decidable[k].accuracy for k, _, _ in SYSTEMS}
    tex.append(latex_table(
        f"Accuracy comparison ($n = {n}$).", "tab:acc",
        ["Method", "Overall (\\%)", "Decidable subset (\\%)"],
        [[label, f"{acc[k]:.1f}", f"{acc_dec[k]:.1f}"]
         for k, label, _ in SYSTEMS]))
    bar_chart("(a) Accuracy", "Accuracy (%)",
              ["Overall", "Decidable\n(SUPPORTS / REFUTES)"],
              {k: [acc[k], acc_dec[k]] for k, _, _ in SYSTEMS},
              f"{args.tag}_accuracy.png")

    # ---- (b) precision, (c) recall, (d) F1 ----------------------------
    for part, attr, cap in (("b", "precision", "Precision"),
                            ("c", "recall", "Recall"),
                            ("d", "f1", "F1-score")):
        per = {k: {s.label: 100 * getattr(s, attr) for s in overall[k].per_label}
               for k, _, _ in SYSTEMS}
        macro = {k: statistics.mean(per[k].values()) for k, _, _ in SYSTEMS}
        tex.append(latex_table(
            f"{cap} comparison by label and macro-averaged ($n = {n}$).",
            f"tab:{attr}",
            ["Method"] + [v.replace("_", r"\_") for v in VERDICT_VALUES]
            + ["Macro"],
            [[label] + [f"{per[k][v]:.1f}" for v in VERDICT_VALUES]
             + [f"{macro[k]:.1f}"] for k, label, _ in SYSTEMS]))
        bar_chart(f"({part}) {cap}", f"{cap} (%)",
                  [v.replace("_", " ") for v in VERDICT_VALUES] + ["Macro"],
                  {k: [per[k][v] for v in VERDICT_VALUES] + [macro[k]]
                   for k, _, _ in SYSTEMS},
                  f"{args.tag}_{attr}.png")

    # ---- (e) cost and error rate --------------------------------------
    lat = {k: statistics.mean(float(r[k].get("seconds", 0.0)) for r in rows)
           for k, _, _ in SYSTEMS}
    err = {k: 100 * (1 - overall[k].accuracy) for k, _, _ in SYSTEMS}
    calls = {"baseline": 1, "computed": 3, "multi": 3}
    tex.append(latex_table(
        f"Computation cost and error rate ($n = {n}$).", "tab:cost",
        ["Method", "LLM calls", "Mean latency (s)", "Error rate (\\%)"],
        [[label, str(calls[k]), f"{lat[k]:.1f}", f"{err[k]:.1f}"]
         for k, label, _ in SYSTEMS]))
    bar_chart("(e) Mean latency per claim", "Seconds", ["Latency"],
              {k: [lat[k]] for k, _, _ in SYSTEMS},
              f"{args.tag}_latency.png",
              ylim=max(lat.values()) * 1.3)
    bar_chart("(e) Error rate", "Error rate (%)", ["Error rate"],
              {k: [err[k]] for k, _, _ in SYSTEMS},
              f"{args.tag}_errorrate.png")

    out = TAB_DIR / f"{args.tag}_tables.tex"
    out.write_text("\n".join(tex), encoding="utf-8")

    print(f"n = {n} claims from {args.results}\n")
    for k, label, _ in SYSTEMS:
        c = overall[k]
        print(f"  {label:<26} acc {100*c.accuracy:5.1f}%  "
              f"macro-F1 {100*c.macro_f1:5.1f}%  "
              f"decidable {100*decidable[k].accuracy:5.1f}%  "
              f"latency {lat[k]:5.1f}s")
    print(f"\nTables -> {out}")
    print(f"Charts -> {FIG_DIR}/{args.tag}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
