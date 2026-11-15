"""Tests for the curated knowledge access layer."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge.facts import get_validated_forecast_facts
from knowledge.retrieval import BusinessKnowledgeRetriever


def test_model_facts():
    facts = get_validated_forecast_facts("model", ROOT)

    assert facts["model_version"] == "point_forecast_v1"
    assert facts["forecast_horizon_days"] == 14
    assert facts["feature_count"] == 31


def test_accuracy_facts():
    facts = get_validated_forecast_facts("accuracy", ROOT)

    assert facts["historical_development_windows"] == 15
    assert 0 < facts["historical_mean_fold_wape"] < 1
    assert 0 < facts["final_holdout_wape"] < 1


def test_business_retrieval():
    retriever = BusinessKnowledgeRetriever(ROOT)

    assert len(retriever.passages) == 23

    context = retriever.build_context(
        "Will our planned discounts affect tomorrow's forecast?",
        top_k=3,
    )

    assert any(
        item["section"] == "Promotions"
        for item in context["evidence"]
    )


def test_document_change_detection(tmp_path):
    original = ROOT / "docs" / "knowledge"
    temporary = tmp_path / "docs" / "knowledge"
    temporary.mkdir(parents=True)

    for filename in (
        "model_card.md",
        "business_glossary.md",
        "forecast_methodology.md",
    ):
        shutil.copy2(original / filename, temporary / filename)

    retriever = BusinessKnowledgeRetriever(tmp_path)

    glossary = temporary / "business_glossary.md"
    glossary.write_text(
        glossary.read_text(encoding="utf-8") + "\nChanged.",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="changed after indexing"):
        retriever.search("What is a sales forecast?")


def test_unsupported_fact_topic():
    with pytest.raises(ValueError, match="Unsupported fact topic"):
        get_validated_forecast_facts("unknown", ROOT)
