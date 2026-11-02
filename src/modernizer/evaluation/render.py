"""Markdown rendering of an evaluation report.

The report text always carries its own caveats and, for a dry run, a banner at both ends, so
that a pasted table cannot be mistaken for a measurement of a language model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import EvaluationReport, ModuleRow, PolicyResult

DRY_RUN_WARNING = (
    "A scripted oracle replayed the answer key instead of calling a model. These numbers test "
    "the harness plumbing only and are NOT results of any language model."
)

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _count_rate(k: int | None, n: int) -> str:
    if k is None:
        return "not run"
    return f"{k}/{n} ({_pct(k / n if n else 0.0)})"


def _with_interval(k: int | None, n: int, interval: tuple[float, float] | None) -> str:
    if k is None or interval is None:
        return "not run"
    return f"{_count_rate(k, n)} [{_pct(interval[0])}, {_pct(interval[1])}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines.extend("| " + " | ".join(cells) + " |" for cells in rows)
    return lines


def _routing_label(result: PolicyResult) -> str:
    spec = result.spec
    parts = [spec.policy]
    if result.threshold_used is not None:
        calibrated = (
            "calibrated threshold" if result.planned_open_share is not None else "threshold"
        )
        parts.append(f"{calibrated} {result.threshold_used:g}")
    parts.append("escalates on failure" if spec.escalate else "no escalation")
    return ", ".join(parts)


def _cost(value: float) -> str:
    return f"${value:.4f}"


def _banner(report: EvaluationReport) -> list[str]:
    if not report.dry_run:
        return []
    return [f"> **DRY RUN - NOT MODEL RESULTS.** {DRY_RUN_WARNING}", ""]


def _settings_lines(report: EvaluationReport) -> list[str]:
    s = report.settings
    rows = [[f"`{key}`", f"`{value}`"] for key, value in s.items()]
    return ["## Settings", "", *_table(["Setting", "Value"], rows), ""]


def _quality_table(report: EvaluationReport) -> list[str]:
    header = [
        "Policy",
        "Routing",
        "Generated-test pass [95% Wilson]",
        "Held-out pass [95% Wilson]",
        "Held-out tests passed (mean)",
        "Routed open",
        "Finished on open and passed generated tests",
        "Escalated",
    ]
    rows = []
    for r in report.policies:
        rows.append(
            [
                r.spec.name,
                _routing_label(r),
                _with_interval(r.generated_pass, r.n, r.generated_interval()),
                _with_interval(r.reference_pass, r.n, r.reference_interval()),
                _pct(r.reference_test_fraction),
                _count_rate(r.routed_open, r.n),
                _count_rate(r.final_open_successes, r.n),
                _count_rate(r.escalated, r.n),
            ]
        )
    lines = ["## Quality", "", *_table(header, rows), ""]
    if not report.run_tests:
        lines += [
            "Tests were not run: the generated-test column only reflects static validation.",
            "",
        ]
    return lines


def _share(count: int, n: int) -> str:
    return f"{count}/{n} ({_pct(count / n if n else 0.0)})"


def _routing_section(report: EvaluationReport) -> list[str]:
    header = [
        "Policy",
        "Target open share (policy parameter)",
        "Planned open share",
        "Routed open",
        "Forced to frontier by feature rules",
        "Final open share (after escalation)",
    ]
    rows = []
    for r in report.policies:
        rows.append(
            [
                r.spec.name,
                "none" if r.target_open_fraction is None else _pct(r.target_open_fraction),
                "n/a" if r.planned_open_share is None else _pct(r.planned_open_share),
                _share(r.routed_open, r.n),
                _share(r.forced_frontier, r.n),
                _share(r.final_open, r.n),
            ]
        )
    return [
        "## Routing",
        "",
        *_table(header, rows),
        "",
        "A target open share is a policy parameter that the routing threshold is calibrated "
        "toward on this batch; it is not a result. The planned share is what the router "
        "computes from the difficulty scores before any model is called, the routed share is "
        "what happened at routing time, and the two can differ from the target when scores "
        "tie or when feature rules force modules to the frontier. Every escalation moves a "
        "module from the open to the frontier tier, so the final open share is never above the "
        "routed share.",
        "",
    ]


def _tier_tokens(result: PolicyResult, tier: str) -> str:
    counts = result.tokens_by_tier.get(tier)
    return f"{counts.total:,}" if counts else "0"


def _cost_table(report: EvaluationReport) -> list[str]:
    header = [
        "Policy",
        "Input tokens",
        "Output tokens",
        "Total tokens",
        "Total excl. test generation",
        "Open-tier tokens",
        "Frontier-tier tokens",
        "Cost (USD)",
        "Cost excl. test generation (USD)",
    ]
    rows = [
        [
            r.spec.name,
            f"{r.tokens.input_tokens:,}",
            f"{r.tokens.output_tokens:,}",
            f"{r.tokens.total:,}",
            f"{r.tokens_excluding_testgen.total:,}",
            _tier_tokens(r, "open"),
            _tier_tokens(r, "frontier"),
            _cost(r.cost_usd_total),
            _cost(r.cost_usd_excluding_testgen),
        ]
        for r in report.policies
    ]
    return [
        "## Tokens and cost",
        "",
        "Totals come from the OpenTelemetry meter and were cross-checked against the usage "
        "ledger, the per-module results and the per-call records. The tier columns include "
        f"test generation, which runs on the {report.settings.get('testgen_tier', 'configured')} "
        "tier whatever the routing policy; use the 'excl. test generation' columns to see "
        "what routing changed.",
        "",
        *_table(header, rows),
        "",
    ]


def _comparison_section(report: EvaluationReport) -> list[str]:
    if not report.comparisons:
        return []
    header = [
        "Policy",
        "Tokens vs baseline",
        "Tokens excl. test gen",
        "Cost vs baseline",
        "Cost excl. test gen",
        "Held-out pass delta (points)",
        "Held-out discordant (baseline only / policy only)",
        "Held-out sign-test p",
    ]
    rows = []
    for c in report.comparisons:
        ref = c.reference
        rows.append(
            [
                c.policy,
                _reduction(c.token_reduction_pct),
                _reduction(c.token_reduction_pct_excluding_testgen),
                _reduction(c.cost_reduction_pct),
                _reduction(c.cost_reduction_pct_excluding_testgen),
                "not run" if ref is None else f"{ref.delta * 100:+.1f}",
                "not run"
                if ref is None
                else f"{ref.only_baseline_passes} / {ref.only_policy_passes}",
                "not run" if ref is None else f"{ref.p_value:.3f}",
            ]
        )
    lines = [
        f"## Comparison against baseline `{report.baseline}`",
        "",
        "Reduction columns are percentages by which the policy is below the baseline; a negative "
        "value means the policy used more.",
        "",
        *_table(header, rows),
        "",
    ]
    for c in report.comparisons:
        lines += [f"**{c.policy}**: {c.verdict}", ""]
    return lines


def _reduction(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:.1f}% lower" if pct >= 0 else f"{-pct:.1f}% higher"


def _difficulty_section(report: EvaluationReport) -> list[str]:
    header = ["Difficulty", "n", "Generated-test pass", "Held-out pass", "Routed open", "Escalated"]
    lines = ["## Per-difficulty breakdown", ""]
    for r in report.policies:
        rows = []
        for label, g in sorted(
            r.by_difficulty().items(), key=lambda kv: _DIFFICULTY_ORDER.get(kv[0], 99)
        ):
            n = g["n"] or 0
            rows.append(
                [
                    label,
                    str(n),
                    _count_rate(g["generated_pass"], n),
                    _count_rate(g["reference_pass"], n),
                    _count_rate(g["routed_open"], n),
                    _count_rate(g["escalated"], n),
                ]
            )
        lines += [f"**{r.spec.name}**", "", *_table(header, rows), ""]
    lines += [
        "Per-difficulty cells hold only a handful of modules each; read them as a listing, "
        "not as rates.",
        "",
    ]
    return lines


def _module_cell(row: ModuleRow) -> str:
    tiers = (
        row.routed_tier
        if row.routed_tier == row.final_tier
        else (f"{row.routed_tier}->{row.final_tier}")
    )
    held_out = (
        "n/a" if row.reference_success is None else "pass" if row.reference_success else "FAIL"
    )
    return f"{tiers}; gen {'pass' if row.generated_success else 'FAIL'}; held-out {held_out}"


def _module_section(report: EvaluationReport) -> list[str]:
    if not report.policies:
        return []
    by_policy = [{row.id: row for row in r.rows} for r in report.policies]
    first = report.policies[0]
    header = ["Module", "Label", "Router score", *[r.spec.name for r in report.policies]]
    rows = [
        [
            row.id,
            row.difficulty,
            f"{row.difficulty_score:.2f}",
            *[_module_cell(rows_by_id[row.id]) for rows_by_id in by_policy],
        ]
        for row in first.rows
    ]
    return ["## Per-module outcomes", "", *_table(header, rows), ""]


def caveats(report: EvaluationReport) -> list[str]:
    """Reasons the numbers must not be over-read; always shown with the report."""
    n = report.corpus.get("n", 0)
    items = [
        f"One run per policy over {n} synthetic benchmark modules at temperature 0. Run-to-run "
        "model variance was not measured, so the intervals understate the real uncertainty.",
        "The Wilson intervals describe sampling noise over this corpus only. They say nothing "
        "about other code bases; the modules were hand-built for this benchmark.",
        "A held-out pass means the module passed these hand-written tests, not that it is "
        "behaviourally equivalent to the legacy code. The tests are not exhaustive.",
        "The mean held-out test fraction gives partial credit: a port with some wrong routines "
        "still passes the tests of the routines it got right. The all-tests pass rate is the "
        "headline metric.",
        "Generated-test success is judged by tests an LLM wrote itself, so it can be too lenient "
        "or too strict. Prefer the held-out column; it is the independent check.",
        "Reduction percentages compare each policy with the baseline arm only. 'No detectable "
        "difference' is absence of evidence, never proof of equal quality, and several "
        "comparisons are made without multiple-comparison correction.",
        "Test generation runs only for candidates that pass static validation, so its spend "
        "still differs between policies. The 'excluding test generation' columns are a view "
        "that isolates conversion cost, not a controlled experiment.",
        "Costs use the repository price table, whose entries are placeholders until replaced "
        "with current list prices; token counts are the price-independent figures.",
        "The open model is self-hosted (Gemma 3 12B in the default configuration). Its "
        "per-token price in the pricing table is an amortised placeholder unless the run "
        "supplied a pricing file (MODERNIZER_PRICING_FILE) built with "
        "`llm.pricing.amortised_price_per_mtok` from the real accelerator price, throughput and "
        "utilisation; a cost reduction against the frontier model is only as good as that "
        "assumption.",
        "Token counts from different models use different tokenizers and are not comparable "
        "one-to-one, so a token reduction between a frontier and an open arm mixes tokenizers; "
        "read it together with the cost figures and their price assumptions.",
        "If the routing threshold was tuned on the same modules it is scored on, the open-model "
        "share and the savings are in-sample and optimistic. Calibrate on one half "
        "(`split_calibration`) and report the other. A target open share calibrates the "
        "threshold on the evaluated batch itself, so it is in-sample by construction.",
    ]
    if not report.run_tests:
        items.append(
            "Tests were disabled for this run: pass columns reflect static validation only."
        )
    items.extend(report.notes)
    return items


def render_markdown(report: EvaluationReport) -> str:
    lines: list[str] = ["# Evaluation report", ""]
    lines += _banner(report)
    by_difficulty = report.corpus.get("by_difficulty", {})
    mix = ", ".join(f"{k}: {v}" for k, v in by_difficulty.items())
    lines += [
        f"Modules: **{report.corpus.get('n', 0)}** ({mix}). Generated at {report.generated_at}.",
        "",
    ]
    lines += _settings_lines(report)
    lines += _quality_table(report)
    lines += _routing_section(report)
    lines += _cost_table(report)
    lines += _comparison_section(report)
    lines += _difficulty_section(report)
    lines += _module_section(report)
    lines += ["## Caveats", "", *[f"- {item}" for item in caveats(report)], ""]
    if report.dry_run:
        lines += [f"> **DRY RUN - NOT MODEL RESULTS.** {DRY_RUN_WARNING}", ""]
    return "\n".join(lines).rstrip("\n") + "\n"
