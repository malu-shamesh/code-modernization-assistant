from __future__ import annotations

import json

import pytest

from modernizer.evaluation import EvaluationReport, compare_results

from .helpers import make_report, make_result, make_row


def test_dry_run_report_carries_banner_and_flag():
    report = make_report(dry_run=True)
    markdown = report.to_markdown()
    data = json.loads(report.to_json())

    assert data["dry_run"] is True
    assert "NOT results of any language model" in data["dry_run_warning"]
    assert markdown.count("DRY RUN - NOT MODEL RESULTS") == 2
    assert markdown.index("DRY RUN") < markdown.index("## Settings")
    assert markdown.rstrip().splitlines()[-1].startswith("> **DRY RUN")
    assert all("DRY RUN" in c["verdict"] for c in data["comparisons"])


def test_real_report_has_no_banner():
    report = make_report(dry_run=False)
    markdown = report.to_markdown()
    data = json.loads(report.to_json())

    assert data["dry_run"] is False
    assert data["dry_run_warning"] is None
    assert "DRY RUN" not in markdown


def test_markdown_has_the_expected_sections():
    markdown = make_report().to_markdown()
    for heading in (
        "## Settings",
        "## Quality",
        "## Tokens and cost",
        "## Comparison against baseline `always_frontier`",
        "## Per-difficulty breakdown",
        "## Per-module outcomes",
        "## Caveats",
    ):
        assert heading in markdown
    assert "Wilson" in markdown
    assert "excl. test generation" in markdown
    assert "easy" in markdown and "medium" in markdown and "hard" in markdown
    assert "absence of evidence" in markdown
    assert "in-sample" in markdown


def test_markdown_never_uses_an_em_dash():
    assert chr(0x2014) not in make_report(dry_run=True).to_markdown()


def test_extra_notes_appear_in_the_caveats():
    report = make_report()
    noted = EvaluationReport(**{**report.__dict__, "notes": ["threshold tuned on half A"]})
    assert "- threshold tuned on half A" in noted.to_markdown()


def test_json_round_trips_losslessly():
    report = make_report(dry_run=True)
    text = report.to_json()
    assert json.loads(text) == report.to_dict()
    restored = EvaluationReport.from_json(text)
    assert restored == report
    assert restored.to_json() == text
    assert restored.to_markdown() == report.to_markdown()


def test_unknown_schema_version_is_rejected():
    data = make_report().to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        EvaluationReport.from_dict(data)


def test_policy_lookup():
    report = make_report()
    assert report.policy("complexity").spec.threshold == 0.5
    with pytest.raises(KeyError):
        report.policy("missing")


def test_cost_table_says_tier_columns_include_test_generation():
    report = make_report()
    markdown = EvaluationReport(**{**report.__dict__, "settings": {"testgen_tier": "frontier"}})
    text = markdown.to_markdown()
    assert "tier columns include test generation" in text
    assert "runs on the frontier tier" in text


def routing_report():
    """A complexity arm with a target of 50% that planned 60%, routed 60% and kept 40% open."""
    labels = ["easy", "medium", "hard"]
    rows = [
        make_row(
            f"m{i}",
            difficulty=labels[i % 3],
            routed="open" if i < 3 else "frontier",
            final="open" if i < 2 else "frontier",
            forced=i == 4,
            tokens=50,
            cost=0.1,
        )
        for i in range(5)
    ]
    baseline = make_result(
        "always_frontier", [make_row(f"m{i}", routed="frontier", tokens=100) for i in range(5)]
    )
    policy = make_result(
        "complexity",
        rows,
        escalate=True,
        threshold=0.42,
        target=0.5,
        planned=0.6,
    )
    report = make_report()
    return EvaluationReport(
        **{
            **report.__dict__,
            "policies": [baseline, policy],
            "comparisons": [compare_results(baseline, policy)],
            "corpus": {"n": 5, "by_difficulty": {"easy": 2, "medium": 2, "hard": 1}},
        }
    )


def test_routing_section_shows_target_planned_routed_forced_and_final_side_by_side():
    report = routing_report()
    complexity = report.policy("complexity")
    assert (complexity.routed_open, complexity.final_open, complexity.forced_frontier) == (3, 2, 1)

    lines = report.to_markdown().splitlines()
    start = lines.index("## Routing")
    table = [line for line in lines[start:] if line.startswith("| ")]
    header = table[0]
    for column in ("Target open share", "Planned open share", "Routed open", "Forced to frontier"):
        assert column in header
    assert "Final open share (after escalation)" in header
    row = next(line for line in table if line.startswith("| complexity "))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[1:] == ["50.0%", "60.0%", "3/5 (60.0%)", "1/5 (20.0%)", "2/5 (40.0%)"]
    baseline_row = next(line for line in table if line.startswith("| always_frontier "))
    assert "none" in baseline_row and "n/a" in baseline_row


def test_routing_section_states_that_a_target_is_not_a_result():
    text = " ".join(routing_report().to_markdown().split())
    assert "A target open share is a policy parameter" in text
    assert "it is not a result" in text
    assert "Every escalation moves a module from the open to the frontier tier" in text
    assert "final open share is never above the routed share" in text


def test_calibrated_threshold_is_labelled_as_such():
    markdown = routing_report().to_markdown()
    assert "complexity, calibrated threshold 0.42, escalates on failure" in markdown
    # A fixed threshold is not called calibrated.
    assert "complexity, threshold 0.5, escalates on failure" in make_report().to_markdown()


def test_caveats_cover_self_hosting_pricing_and_tokenizers():
    text = " ".join(make_report().to_markdown().split())
    assert "self-hosted (Gemma 3 12B" in text
    assert "amortised placeholder" in text
    assert "MODERNIZER_PRICING_FILE" in text
    assert "llm.pricing.amortised_price_per_mtok" in text
    assert "different tokenizers" in text and "not comparable one-to-one" in text
    assert "in-sample by construction" in text


def test_no_python_2_wording_is_left_in_the_report():
    text = routing_report().to_markdown().lower()
    assert "python 2" not in text and "py2" not in text


def test_row_without_forced_flag_defaults_to_not_forced():
    assert make_row("x").forced_frontier is False
