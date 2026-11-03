"""Source-level regressions for professional relevance, independent of labels."""
import json
from pathlib import Path

import numpy as np
import pytest

from ML2.features import FEATURE_NAMES, extract_features, roles, uncertain_skill_mention
from ML2.ranker import LinearRanker, SCORE_POLICY_VERSION


DATA = {"title": "Middle Data Engineer", "experience": "От 3 до 6 лет",
        "requirements": "SAP, Tableau", "responsibilities": "ETL data pipelines"}
BACKEND = {"title": "Middle Backend Developer", "experience": "От 3 до 6 лет",
           "requirements": "Python, FastAPI, PostgreSQL"}


def signals(text, vacancy):
    vector, facts = extract_features(text, vacancy)
    return dict(zip(FEATURE_NAMES, vector)), facts


@pytest.mark.parametrize("profile", ["Middle frontend developer", "Middle Python backend developer", "Middle fullstack developer", "Junior Python backend developer"])
def test_unrelated_years_and_level_do_not_manufacture_data_engineering(profile):
    features, facts = signals(profile + ". Work experience: 5 years. Git, REST API.", DATA)
    assert features["experience_compatibility"] == 0
    assert features["seniority_compatibility"] == 0
    assert features["role_compatibility"] == 0
    assert facts["relevant_years"] is None
    assert not facts["relevance_supported"]


def test_bare_same_title_and_total_still_need_technical_evidence():
    features, facts = signals("Middle Data Engineer. Work experience: 5 years. Git REST API.", DATA)
    assert features["experience_compatibility"] == 0
    assert features["seniority_compatibility"] == 0
    assert not facts["relevance_supported"]


def test_specific_data_professions_are_not_interchangeable():
    assert roles("Data Analyst") == {"data_analysis"}
    assert roles("Data Scientist") == {"data_science"}
    assert roles("Data Engineer") == {"data_engineering"}
    features, facts = signals("Middle Data Analyst. Work experience: 5 years. SAP Tableau.", DATA)
    assert features["skill_coverage"] == pytest.approx(2 / 3)
    assert features["experience_compatibility"] == 0
    assert facts["relevant_years"] is None


@pytest.mark.parametrize("text", [
    "Senior frontend developer 5 years, Junior backend developer 1 year. Python FastAPI PostgreSQL.",
    "5 years total experience, including 1 year as Junior backend developer. Previously frontend developer. Python FastAPI PostgreSQL.",
    "Общий опыт 5 лет, из них Junior backend developer 1 год. Ранее frontend developer. Python FastAPI PostgreSQL.",
    "Frontend developer: 5 years; Junior backend developer: 1 year. Python FastAPI PostgreSQL.",
])
def test_mixed_career_uses_only_explicit_backend_year_and_level(text):
    features, facts = signals(text, BACKEND)
    assert facts["relevant_years"] == 1
    assert features["experience_compatibility"] == pytest.approx(-2 / 3)
    assert features["seniority_compatibility"] == -.25
    assert facts["experience_evidence"]


def test_explicit_scope_after_total_style_prefix_is_retained():
    _, facts = signals("Frontend developer. Work experience: 2 years in backend. Python FastAPI PostgreSQL.", BACKEND)
    assert facts["relevant_years"] == 2


@pytest.mark.parametrize("text", [
    "Frontend developer. Work experience: 5 years. Хочу стать Data Engineer, изучаю SAP и Tableau.",
    "Middle Data Engineer. Work experience: 5 years. Never built ETL data pipelines with SAP or Tableau.",
    "Middle Data Engineer. Work experience: 5 years. No experience with SAP and Tableau or ETL data pipelines.",
    "Middle Data Engineer. Work experience: 5 years. I want to learn SAP and Tableau.",
])
def test_negative_and_aspirational_claims_are_not_positive_evidence(text):
    features, facts = signals(text, DATA)
    assert features["skill_coverage"] == 0
    assert features["task_coverage"] == 0
    assert features["experience_compatibility"] == 0
    assert features["seniority_compatibility"] == 0
    assert not facts["relevance_supported"]


def test_skill_support_controls_scoped_experience_instead_of_total_only():
    partial, _ = signals("Middle backend developer. Work experience: 5 years. Python.", BACKEND)
    full, _ = signals("Middle backend developer. Work experience: 5 years. Python FastAPI PostgreSQL.", BACKEND)
    assert partial["experience_compatibility"] == pytest.approx(1 / 3)
    assert full["experience_compatibility"] == 1
    assert partial["seniority_compatibility"] < full["seniority_compatibility"]


def test_overall_mixed_total_without_role_scoped_years_stays_unknown():
    features, facts = signals("Frontend developer and backend developer. Work experience: 6 years. Python FastAPI PostgreSQL.", BACKEND)
    assert facts["claimed_total_years"] == 6
    assert facts["relevant_years"] is None
    assert features["experience_compatibility"] == 0


def test_unreasonable_durations_are_unknown_and_finite():
    features, facts = signals("Middle backend developer. Work experience: " + "9" * 400 + " years. Python FastAPI PostgreSQL.", BACKEND)
    assert facts["claimed_total_years"] is None
    assert facts["relevant_years"] is None
    assert all(np.isfinite(list(features.values())))


def test_requirement_evidence_does_not_borrow_optional_skill_quote():
    _, facts = signals("Backend developer. Python. Tableau.", {**BACKEND, "requirements": "Mandatory Python. Optional Tableau."})
    evidence = facts["feature_evidence"]["mandatory_skill_confirmation"]
    assert evidence and all("python" in row["quote"] for row in evidence)
    assert not uncertain_skill_mention("Machine Learning models", "machine learning")
    assert uncertain_skill_mention("I want to learn Python and FastAPI", "python")


def test_additive_explanation_equals_prediction_and_zero_evidence_is_zero():
    rows = [{"cv_id": "a", "relevance": 0, "overall_score": 1},
            {"cv_id": "a", "relevance": 2, "overall_score": 3}]
    x = np.zeros((2, len(FEATURE_NAMES)))
    x[1, 0] = 1
    model = LinearRanker.fit(rows, x)
    reports = model.explain(x)
    assert model.predict(np.zeros((1, len(FEATURE_NAMES))))[0] == 0
    assert np.all(model.mean == 0)
    for score, report in zip(model.predict(x), reports):
        assert len(report["contributions"]) == len(FEATURE_NAMES)
        assert report["raw_score"] == pytest.approx(score)
        assert report["base_score"] + sum(row["contribution"] for row in report["contributions"]) == pytest.approx(score)


def test_shipped_model_rewards_real_data_evidence_not_unrelated_middle():
    model = LinearRanker.load(Path(__file__).resolve().parents[2] / "backend/models/ranker.json")
    relevant = extract_features("Middle Data Engineer. Work experience: 4 years. Built ETL data pipelines with SAP and Tableau.", DATA)[0]
    unrelated = extract_features("Middle frontend developer. Work experience: 5 years. React TypeScript Git REST API.", DATA)[0]
    scores = model.predict([relevant, unrelated])
    assert scores[0] > 0
    assert scores[1] == 0
    assert scores[0] > scores[1]


def test_explicit_experience_shortfall_adjusts_only_positive_base():
    weights = np.zeros(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("skill_coverage")] = 1
    model = LinearRanker(np.zeros(len(weights)), np.ones(len(weights)), weights)
    matrix = np.zeros((3, len(weights)))
    matrix[:, FEATURE_NAMES.index("skill_coverage")] = 1
    matrix[:, FEATURE_NAMES.index("role_compatibility")] = 1
    matrix[0, FEATURE_NAMES.index("experience_compatibility")] = -2 / 3
    matrix[1, FEATURE_NAMES.index("experience_compatibility")] = 1
    # Row 2 is unknown or no minimum, not fabricated negative experience.
    np.testing.assert_allclose(model.predict(matrix), [1 / 3, 1, 1])
    reports = model.explain(matrix)
    assert reports[0]["adjustments"][0]["policy"] == SCORE_POLICY_VERSION
    assert reports[0]["adjustments"][0]["factor"] == pytest.approx(1 / 3)
    assert reports[1]["adjustments"] == reports[2]["adjustments"] == []
    for score, report in zip(model.predict(matrix), reports):
        assert report["raw_score"] == pytest.approx(1)
        assert report["final_score"] == pytest.approx(score)
        assert report["base_score"] + sum(r["contribution"] for r in report["contributions"]) + sum(r["contribution"] for r in report["adjustments"]) == pytest.approx(score)


def test_experience_policy_does_not_raise_a_negative_linear_score():
    weights = np.zeros(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("experience_compatibility")] = 1
    model = LinearRanker(np.zeros(len(weights)), np.ones(len(weights)), weights)
    vector = np.zeros(len(weights))
    vector[FEATURE_NAMES.index("role_compatibility")] = 1
    vector[FEATURE_NAMES.index("experience_compatibility")] = -2 / 3
    report = model.explain([vector])[0]
    assert report["raw_score"] == report["final_score"] == pytest.approx(-2 / 3)
    assert report["adjustments"][0]["contribution"] == 0
    assert model.predict([vector])[0] == pytest.approx(-2 / 3)


def test_shipped_policy_distinguishes_one_and_three_relevant_years():
    model = LinearRanker.load(Path(__file__).resolve().parents[2] / "backend/models/ranker.json")
    texts = [f"Middle Data Engineer. Work experience: {years} years. Built ETL data pipelines with SAP and Tableau." for years in (1, 3)]
    vectors = [extract_features(text, DATA)[0] for text in texts]
    short, sufficient = model.explain(vectors)
    assert short["final_score"] == pytest.approx(short["raw_score"] / 3)
    assert sufficient["final_score"] == sufficient["raw_score"]
    assert short["final_score"] < sufficient["final_score"]


def test_cli_rerank_adapter_reconciles_raw_features_policy_and_final_score():
    weights = np.zeros(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("skill_coverage")] = 1
    model = LinearRanker(np.zeros(len(weights)), np.ones(len(weights)), weights)
    result = model.rerank("Middle Backend Developer. Work experience: 1 year. Python.",
                          [{"vacancy_id": "backend", "title": "Middle Backend Developer",
                            "requirements": "Python", "experience": "От 3 до 6 лет"}])[0]
    breakdown = result["ml2_score_breakdown"]
    assert breakdown["score_policy_version"] == SCORE_POLICY_VERSION
    assert result["ml2_score"] == pytest.approx(1 / 3)
    assert breakdown["raw_score"] == 1
    assert breakdown["final_score"] == result["ml2_score"]
    assert sum(result["ml2_contributions"].values()) + sum(row["contribution"] for row in breakdown["adjustments"]) == pytest.approx(result["ml2_score"])


@pytest.mark.parametrize("remove_top", [True, False])
def test_saved_model_cannot_silently_use_another_score_policy(tmp_path, remove_top):
    weights = np.zeros(len(FEATURE_NAMES))
    model = LinearRanker(np.zeros(len(weights)), np.ones(len(weights)), weights)
    path = tmp_path / "model.json"
    model.save(path)
    saved = json.loads(path.read_text())
    if remove_top:
        saved.pop("score_policy_version")
    else:
        saved["metadata"]["score_policy_version"] = "linear-only"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Incompatible"):
        LinearRanker.load(path)
