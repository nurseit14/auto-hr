"""Mechanical checks for a frozen synthetic evaluation, not model-score targets."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ML2.hr_benchmark import DEFAULT_FIXTURE, evaluate_hr_benchmark, file_sha256, load_cases


FIXTURE_SHA256 = "f730bd256cd915e5caa0ed8fce0777b4a822ab232b80ef1a53fd40e9cf694a28"
SECOND_FIXTURE_SHA256 = "7cd71500656d31b8d82ce8aa181e67d1352604b03ab3bf24ca7051cd25b301d6"


class Scores:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, matrix):
        assert len(matrix) == len(self.scores)
        return np.array(self.scores)

    def to_dict(self):
        return {"model_version": "synthetic-mechanical-stub", "feature_version": "stub"}


def small_fixture(tmp_path):
    fixture = copy.deepcopy(load_cases(DEFAULT_FIXTURE))
    fixture["cases"] = fixture["cases"][:1]
    fixture["cases"][0]["candidates"] = fixture["cases"][0]["candidates"][:3]
    fixture["cases"][0]["preferences"] = [
        {"higher": "strong", "lower": "partial"},
        {"higher": "partial", "lower": "junior"},
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_evaluation_fixture_is_frozen_and_explicitly_synthetic():
    fixture = load_cases(DEFAULT_FIXTURE)
    assert file_sha256(DEFAULT_FIXTURE) == FIXTURE_SHA256
    assert len(fixture["cases"]) == 10
    assert sum(len(case["candidates"]) for case in fixture["cases"]) == 60
    assert "engineering" in fixture["provenance"].lower()
    assert "not" in fixture["label_policy"].lower()


def test_perfect_and_reversed_candidate_orders_are_counted_per_vacancy(tmp_path):
    path = small_fixture(tmp_path)
    good, details = evaluate_hr_benchmark(Scores([3, 2, 1]), path)
    bad, _ = evaluate_hr_benchmark(Scores([1, 2, 3]), path)
    assert good["metrics"]["queries"] == 1
    assert good["metrics"]["ndcg_at_5"] == 1
    assert good["metrics"]["pairwise_accuracy"] == 1
    assert good["predeclared_preferences"]["strict_correct"] == 2
    assert good["predeclared_preferences"]["macro_credit"] == 1
    assert bad["metrics"]["ndcg_at_5"] < 1
    assert bad["metrics"]["pairwise_accuracy"] == 0
    assert bad["predeclared_preferences"]["incorrect"] == 2
    assert len(details["case_scores"]) == 1
    assert "details" not in good  # Aggregate report stays free of case scores.


def test_ties_receive_half_preference_credit_and_nonfinite_scores_fail(tmp_path):
    path = small_fixture(tmp_path)
    report, _ = evaluate_hr_benchmark(Scores([0, 0, 0]), path)
    assert report["metrics"]["pairwise_accuracy"] == .5
    assert report["predeclared_preferences"]["strict_correct"] == 0
    assert report["predeclared_preferences"]["ties"] == 2
    assert report["predeclared_preferences"]["macro_credit"] == .5
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_hr_benchmark(Scores([1, float("nan"), 0]), path)


def test_unknown_preference_target_is_rejected(tmp_path):
    path = small_fixture(tmp_path)
    fixture = json.loads(path.read_text())
    fixture["cases"][0]["preferences"][0]["higher"] = "not-a-candidate"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown candidate"):
        load_cases(path)


def test_second_engineering_fixture_is_frozen_with_disclosed_context():
    path = Path(DEFAULT_FIXTURE).with_name("hr_ranking_cases_v2.json")
    fixture = load_cases(path)
    assert file_sha256(path) == SECOND_FIXTURE_SHA256
    assert len(fixture["cases"]) == 8
    assert sum(len(case["candidates"]) for case in fixture["cases"]) == 48
    assert "initial evaluation" in fixture["provenance"].lower()
    assert "author knew" in fixture["usage"].lower()
