from __future__ import annotations

import logging
import math
import os
from importlib import resources

import pytest

from modernizer.llm import PricingTable, amortised_price_per_mtok, cost_usd
from modernizer.llm.pricing import ModelPrice
from modernizer.models import TokenUsage

YAML = """
models:
  alpha:
    input_per_mtok: 1.0
    output_per_mtok: 2.0
  alpha-pro:
    input_per_mtok: 10.0
    output_per_mtok: 20.0
"""


@pytest.fixture(autouse=True)
def no_pricing_override(monkeypatch):
    monkeypatch.delenv("MODERNIZER_PRICING_FILE", raising=False)


@pytest.fixture
def table(tmp_path) -> PricingTable:
    path = tmp_path / "prices.yaml"
    path.write_text(YAML, encoding="utf-8")
    return PricingTable.from_yaml(path)


def test_cost_is_per_million_tokens(table):
    usage = TokenUsage(input_tokens=500_000, output_tokens=250_000)
    assert table.cost_usd("alpha", usage) == pytest.approx(0.5 + 0.5)


def test_longest_prefix_wins(table):
    usage = TokenUsage(1_000_000, 0)
    assert table.cost_usd("alpha-pro-preview-0101", usage) == pytest.approx(10.0)
    assert table.cost_usd("alpha-mini", usage) == pytest.approx(1.0)


def test_lookup_is_case_insensitive_and_strips_resource_prefix(table):
    usage = TokenUsage(1_000_000, 0)
    assert table.cost_usd("ALPHA", usage) == pytest.approx(1.0)
    assert table.cost_usd("publishers/google/models/alpha", usage) == pytest.approx(1.0)


def test_unknown_model_costs_zero_and_warns_once(table, caplog):
    usage = TokenUsage(10, 10)
    with caplog.at_level(logging.WARNING, logger="modernizer.llm.pricing"):
        assert table.cost_usd("mystery", usage) == 0.0
        assert table.cost_usd("mystery", usage) == 0.0
        assert table.cost_usd("other", usage) == 0.0
    messages = [r.getMessage() for r in caplog.records]
    assert sum("'mystery'" in m for m in messages) == 1
    assert sum("'other'" in m for m in messages) == 1


def test_contains(table):
    assert "alpha" in table
    assert "zzz" not in table


@pytest.mark.parametrize(
    "text",
    [
        "not: a table",
        "models: [1, 2]",
        "models:\n  a: 3",
        "models:\n  a: {input_per_mtok: 1}",
        "models:\n  a: {input_per_mtok: x, output_per_mtok: 1}",
        "models:\n  a: {input_per_mtok: -1, output_per_mtok: 1}",
    ],
)
def test_malformed_yaml_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        PricingTable.from_yaml(path)


def test_duplicate_entries_after_normalisation_rejected():
    price = ModelPrice(1.0, 1.0)
    with pytest.raises(ValueError):
        PricingTable({"A": price, "a": price})


def test_packaged_default_covers_default_models():
    packaged = PricingTable.default()
    models = ("gemini-2.5-pro", "gemma-3-12b-it", "qwen/qwen3-coder-480b-a35b-instruct-maas")
    for model in (*models, "mock-open"):
        assert model in packaged
    assert cost_usd("gemini-2.5-pro", TokenUsage(1000, 1000)) > 0
    assert cost_usd("mock-frontier", TokenUsage(1000, 1000)) == 0.0


def test_packaged_gemma_entry_is_flagged_as_a_placeholder():
    text = resources.files("modernizer.llm").joinpath("pricing.yaml").read_text(encoding="utf-8")
    assert "EFFECTIVE AMORTISED RATE" in text
    assert "amortised_price_per_mtok" in text


def test_amortised_price_formula():
    # 3.60 USD/h at 1000 tok/s is 3.6M tokens/h -> 1.00 USD per million tokens
    assert amortised_price_per_mtok(3.60, 1000) == pytest.approx(1.0)
    assert amortised_price_per_mtok(3.60, 1000, utilisation=0.25) == pytest.approx(4.0)
    assert amortised_price_per_mtok(0.0, 500) == 0.0


@pytest.mark.parametrize(
    ("args", "name"),
    [
        ((-1.0, 100.0), "node_hour_usd"),
        ((1.0, 0.0), "tokens_per_second"),
        ((1.0, -5.0), "tokens_per_second"),
        ((1.0, 100.0, 0.0), "utilisation"),
        ((1.0, 100.0, 1.5), "utilisation"),
        ((math.nan, 100.0), "node_hour_usd"),
        ((1.0, math.inf), "tokens_per_second"),
    ],
)
def test_amortised_price_rejects_invalid_inputs(args, name):
    with pytest.raises(ValueError, match=name):
        amortised_price_per_mtok(*args)


def test_pricing_file_env_overrides_packaged_default(tmp_path, monkeypatch):
    path = tmp_path / "private.yaml"
    path.write_text(YAML, encoding="utf-8")
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", str(path))
    assert "alpha" in PricingTable.default()
    assert "gemini-2.5-pro" not in PricingTable.default()
    assert PricingTable.default() is PricingTable.default()
    assert cost_usd("alpha", TokenUsage(1_000_000, 0)) == pytest.approx(1.0)


def test_pricing_file_env_unset_or_blank_uses_packaged_table(monkeypatch):
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", "  ")
    assert "gemma-3-12b-it" in PricingTable.default()
    monkeypatch.delenv("MODERNIZER_PRICING_FILE")
    assert "gemma-3-12b-it" in PricingTable.default()


def test_pricing_file_env_missing_file_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError):
        PricingTable.default()


def test_pricing_file_env_edit_is_picked_up(tmp_path, monkeypatch):
    path = tmp_path / "private.yaml"
    path.write_text(YAML, encoding="utf-8")
    monkeypatch.setenv("MODERNIZER_PRICING_FILE", str(path))
    assert cost_usd("alpha", TokenUsage(1_000_000, 0)) == pytest.approx(1.0)
    path.write_text(YAML.replace("input_per_mtok: 1.0", "input_per_mtok: 7.0"), encoding="utf-8")
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10_000_000))
    assert cost_usd("alpha", TokenUsage(1_000_000, 0)) == pytest.approx(7.0)
