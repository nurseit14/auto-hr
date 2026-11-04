"""Behavioral contract for the fixed ML2-04 control ranker, using fictional data."""

import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ML2.baseline_ranker import BASELINE_VERSION, BASELINE_WEIGHTS, TransparentBaseline


ROOT = Path(__file__).resolve().parents[2]
CV = ("Middle backend developer — 3 years. "
      "Built Python FastAPI services with PostgreSQL. English B2.")


def vacancy(vid="backend", **changes):
    return {
        "vacancy_id": vid,
        "title": "Middle Backend Developer",
        "requirements": "Python, FastAPI, PostgreSQL. English B2.",
        "experience": "От 3 до 6 лет",
        "skill_score": .7,
        "semantic_score": .45,
        "hybrid_score": .6,
        "matching_metadata": {"fixture": True, "config": {"alpha": .4, "tags": ["synthetic"]}},
        **changes,
    }


def score(text=CV, row=None):
    return TransparentBaseline().rerank(text, [vacancy() if row is None else row])[0]


def components(row):
    return {item["key"]: item for item in row["baseline_breakdown"]["components"]}


def requirement(row, skill):
    return next(item for item in row["baseline_evidence"]["requirements"] if item["skill"] == skill)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "ML2.baseline_rerank", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=20,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def test_score_exposes_exact_fixed_contributions_without_losing_ml1_fields():
    source = [vacancy("z"), vacancy("a", hybrid_score=.3)]
    before = copy.deepcopy(source)
    ranked = TransparentBaseline().rerank(CV, source)
    assert source == before
    originals = {row["vacancy_id"]: row for row in before}
    assert len(ranked) == len(source)
    assert [row["baseline_rank"] for row in ranked] == [1, 2]
    for row in ranked:
        for key, value in originals[row["vacancy_id"]].items():
            assert row[key] == value
        report = row["baseline_breakdown"]
        parts = components(row)
        assert set(parts) == set(BASELINE_WEIGHTS)
        assert row["baseline_version"] == BASELINE_VERSION
        assert report["total"] == row["baseline_score"] == math.fsum(
            part["contribution"] for part in parts.values())
        for name, part in parts.items():
            assert part["weight"] == BASELINE_WEIGHTS[name]
            assert part["contribution"] == part["weight"] * part["value"]
            assert part["reason"] and part["status"]
            for proof in part["evidence"]:
                if proof["source"] == "resume":
                    assert proof["quote"] in CV
        assert parts["ml1_hybrid"]["contribution"] == .25 * row["hybrid_score"]
        assert report["feature_version"] and report["feature_fingerprint"]
        assert report["config_fingerprint"] and report["weights_source"]
    ranked[0]["matching_metadata"]["config"]["tags"].append("result-only")
    assert source == before, "Output metadata must not alias the caller's input"


def test_ml1_hybrid_is_actually_used_and_its_original_rank_is_recoverable():
    ranked = TransparentBaseline().rerank(CV, [vacancy("first", hybrid_score=.2),
                                             vacancy("second", hybrid_score=.6)])
    assert [row["vacancy_id"] for row in ranked] == ["second", "first"]
    assert [row["ml1_rank"] for row in ranked] == [2, 1]
    assert ranked[0]["baseline_score"] - ranked[1]["baseline_score"] == pytest.approx(.25 * .4)


def test_exact_ties_keep_input_order_instead_of_sorting_by_vacancy_id():
    source = [vacancy("z-job"), vacancy("a-job"), vacancy("m-job")]
    ranked = TransparentBaseline().rerank(CV, source, top_k=2)
    assert [row["vacancy_id"] for row in ranked] == ["z-job", "a-job"]
    assert [row["ml1_rank"] for row in ranked] == [1, 2]
    assert ranked[0]["baseline_score"] == ranked[1]["baseline_score"]
    assert TransparentBaseline().rerank(CV, []) == []


@pytest.mark.parametrize("value", [None, True, False, "0.5", float("nan"),
                                  float("inf"), -float("inf"), -.1, 1.1])
def test_missing_or_invalid_ml1_score_is_not_silently_invented(value):
    with pytest.raises(ValueError):
        score(row=vacancy(hybrid_score=value))
    missing = vacancy()
    missing.pop("hybrid_score")
    with pytest.raises(ValueError):
        score(row=missing)


@pytest.mark.parametrize("field", ["skill_score", "semantic_score"])
@pytest.mark.parametrize("value", [True, "0.4", float("nan"), 2])
def test_optional_ml1_scores_are_checked_when_present(field, value):
    with pytest.raises(ValueError):
        score(row=vacancy(**{field: value}))


@pytest.mark.parametrize("value", [None, True, "", "   ", float("nan")])
def test_invalid_or_duplicate_ids_are_rejected(value):
    with pytest.raises(ValueError):
        score(row=vacancy(value))
    with pytest.raises(ValueError):
        TransparentBaseline().rerank(CV, [vacancy("same"), vacancy("same")])


def test_id_alias_and_absent_optional_scores_preserve_the_original_record():
    row = vacancy()
    row.pop("vacancy_id")
    row.pop("skill_score")
    row.pop("semantic_score")
    row["id"] = "original-id"
    result = score(row=row)
    assert result["id"] == "original-id"
    assert result["vacancy_id"] == "original-id"
    assert "semantic_score" not in result and "skill_score" not in result


def test_canonical_duplicate_ids_and_reserved_output_fields_are_rejected():
    with pytest.raises(ValueError):
        TransparentBaseline().rerank(CV, [vacancy(1), vacancy("1")])
    with pytest.raises(ValueError):
        score(row=vacancy(baseline_score=.99))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5", None])
def test_invalid_top_k_is_rejected(value):
    with pytest.raises(ValueError):
        TransparentBaseline().rerank(CV, [vacancy()], top_k=value)


@pytest.mark.parametrize("value", ["", "  \n", None, 123])
def test_a_real_source_cv_is_required(value):
    with pytest.raises(ValueError):
        score(text=value)


@pytest.mark.parametrize("value", [None, {}, [None], ["vacancy"]])
def test_invalid_shortlist_structure_is_rejected(value):
    with pytest.raises(ValueError):
        TransparentBaseline().rerank(CV, value)


def test_skill_absence_and_explicit_nonuse_remain_distinct():
    row = vacancy(requirements="Python, Docker.", experience="")
    absent = score("Backend developer. Used Python.", row)
    denied = score("Backend developer. Used Python. Have not used Docker.", row)
    assert requirement(absent, "docker")["status"] == "unknown"
    denial = requirement(denied, "docker")
    assert denial["status"] == "mismatch"
    assert any("docker" in item.get("quote", "").lower() for item in denial["evidence"])


@pytest.mark.parametrize("claim", ["Python не использую.", "Опыта с Python нет."])
def test_direct_present_or_postfix_nonuse_is_not_positive_skill_evidence(claim):
    result = score("Backend developer. " + claim,
                   vacancy(requirements="Python.", experience=""))
    assert requirement(result, "python")["status"] == "mismatch"
    assert components(result)["required_skills"]["contribution"] == 0
    assert components(result)["demonstrated_skills"]["contribution"] == 0
    assert components(result)["professional_profile"]["contribution"] == 0


@pytest.mark.parametrize("claim", ["I want to learn Docker.", "Изучаю Docker."])
def test_learning_is_not_confused_with_confirmed_nonuse_or_work(claim):
    row = vacancy(requirements="Python, Docker.", experience="")
    result = score("Backend developer. Used Python. " + claim, row)
    assert requirement(result, "docker")["status"] == "unknown"


def test_conflicting_skill_claims_do_not_become_positive_confirmation():
    row = vacancy(requirements="Python, Docker.", experience="")
    result = score("Backend developer. Used Python. Built services with Docker. "
                   "Have not used Docker.", row)
    assert requirement(result, "docker")["status"] == "unknown"
    assert result["baseline_evidence"]["warnings"]


def test_explicit_relevant_shortfall_and_unknown_years_have_different_effects():
    short = score("Middle backend developer — 1 year. Built Python FastAPI services with PostgreSQL.")
    unknown = score("Middle backend developer. Built Python FastAPI services with PostgreSQL.")
    assert short["baseline_evidence"]["experience"]["status"] == "mismatch"
    assert short["baseline_evidence"]["experience"]["relevant_years"] == 1
    assert components(short)["relevant_experience"]["value"] < 0
    assert unknown["baseline_evidence"]["experience"]["status"] == "unknown"
    assert unknown["baseline_evidence"]["experience"]["relevant_years"] is None
    assert components(unknown)["relevant_experience"]["contribution"] == 0


def test_general_years_and_middle_level_do_not_boost_an_unrelated_vacancy():
    data = vacancy(title="Middle Data Engineer", requirements="SAP, Tableau.",
                   responsibilities="Build ETL data pipelines.")
    plain = score("Backend developer. Python FastAPI PostgreSQL.", data)
    senior = score("Senior backend developer. Work experience: 12 years. Python FastAPI PostgreSQL.", data)
    assert senior["baseline_evidence"]["experience"]["relevant_years"] is None
    assert components(senior)["relevant_experience"]["contribution"] == 0
    assert components(senior)["professional_profile"]["contribution"] == 0
    assert senior["baseline_score"] == plain["baseline_score"]


def test_months_are_not_misread_as_a_multi_year_experience_shortfall():
    row = vacancy(requirements="Python.", experience="От 6 месяцев")
    result = score("Backend developer — 1 year. Built services with Python.", row)
    experience = result["baseline_evidence"]["experience"]
    assert experience["status"] == "unknown"
    assert experience["required_years"] is None
    assert components(result)["relevant_experience"]["contribution"] == 0
    assert result["baseline_evidence"]["warnings"]


@pytest.mark.parametrize("experience", ["До 5 лет", "Up to 5 years", "Не более 5 лет", "At most 5 years"])
def test_upper_experience_boundary_is_not_invented_as_a_minimum(experience):
    result = score(CV, vacancy(experience=experience))
    requirement = result["baseline_evidence"]["experience"]
    assert requirement["status"] == "unknown"
    assert requirement["required_years"] is None
    assert components(result)["relevant_experience"]["contribution"] == 0


def test_an_explicit_lower_range_boundary_remains_inclusive():
    result = score("Backend developer — 1 year. Built Python FastAPI services with PostgreSQL.",
                   vacancy(experience="От 1 до 3 лет"))
    assert result["baseline_evidence"]["experience"]["status"] == "match"
    assert result["baseline_evidence"]["experience"]["required_years"] == 1


def test_language_mismatch_is_distinct_from_an_unknown_language_level():
    low = score("Backend developer. Built Python FastAPI services with PostgreSQL. English A2.")
    unknown = score("Backend developer. Built Python FastAPI services with PostgreSQL.")
    low_language = next(item for item in low["baseline_evidence"]["languages"]["requirements"]
                        if item["language"] == "english")
    unknown_language = next(item for item in unknown["baseline_evidence"]["languages"]["requirements"]
                            if item["language"] == "english")
    assert low_language["status"] == "mismatch"
    assert unknown_language["status"] == "unknown"
    assert components(low)["languages"]["contribution"] < 0
    assert components(unknown)["languages"]["contribution"] == 0


@pytest.mark.parametrize("claim", ["I want to learn English B2.", "English B2 is my target."])
def test_language_targets_do_not_confirm_an_attained_level(claim):
    row = vacancy(requirements="Python. English B2.", experience="")
    result = score("Backend developer. Built services with Python. " + claim, row)
    language = next(item for item in result["baseline_evidence"]["languages"]["requirements"]
                    if item["language"] == "english")
    assert language["status"] == "unknown"
    assert language["candidate_level"] is None
    assert components(result)["languages"]["contribution"] == 0


def test_a_denied_language_level_is_not_replaced_with_a_fabricated_zero_level():
    row = vacancy(requirements="Python. English B2.", experience="")
    result = score("Backend developer. Built services with Python. English B2 не владею.", row)
    language = result["baseline_evidence"]["languages"]["requirements"][0]
    assert language["status"] == "mismatch"
    assert language["candidate_level"] is None
    assert components(result)["languages"]["contribution"] == 0


@pytest.mark.parametrize("second_language", ["Russian", "Kazakh"])
def test_unknown_requested_language_is_retained_in_language_coverage(second_language):
    row = vacancy(requirements=f"Python. English B2. {second_language} B2.", experience="")
    partial = score("Backend developer. Used Python. English B2.", row)
    complete = score(f"Backend developer. Used Python. English B2. {second_language} B2.", row)
    assert len(partial["baseline_evidence"]["languages"]["requirements"]) == 2
    assert components(partial)["languages"]["value"] == pytest.approx(
        components(complete)["languages"]["value"] / 2)


def test_optional_skills_do_not_dilute_required_skill_component():
    plain = score("Backend developer. Used Python.", vacancy(requirements="Python.", experience=""))
    optional = score("Backend developer. Used Python.", vacancy(requirements="Python. Optional React.", experience=""))
    assert [item["skill"] for item in optional["baseline_evidence"]["requirements"]] == ["python"]
    assert optional["baseline_score"] == plain["baseline_score"]


def test_described_work_and_a_skill_list_supply_different_evidence():
    listed = score("Backend developer. Python FastAPI PostgreSQL.")
    worked = score("Backend developer. Built Python FastAPI services with PostgreSQL.")
    assert components(listed)["required_skills"]["value"] == components(worked)["required_skills"]["value"]
    assert components(worked)["demonstrated_skills"]["value"] > components(listed)["demonstrated_skills"]["value"]


def test_identity_fields_and_annotation_answers_are_not_ranking_features():
    base = score()
    personal = "\nEmail: person@example.com\nNationality: SQL\nГражданство: Pythonland\nЖенщина, родилась 1 января 2000"
    poison = vacancy(overall_score=3, overall_explanation="Senior SAP expert", notes="No Python",
                     predictions=[{"score": -500}], annotations=[{"result": "Docker"}])
    changed = score(CV + personal, poison)
    assert changed["baseline_score"] == base["baseline_score"]
    assert {key: item["value"] for key, item in components(changed).items()} == {
        key: item["value"] for key, item in components(base).items()}


@pytest.mark.parametrize("field", ["title", "requirements", "responsibilities", "text", "description", "experience"])
def test_serialized_annotation_objects_are_not_accepted_as_source_text(field):
    with pytest.raises(ValueError):
        score(row=vacancy(**{field: [{"requirement": "Python", "score": 3}]}))


def test_nested_original_source_takes_precedence_over_outer_annotation_fields():
    original = vacancy()
    nested = {"vacancy_id": original["vacancy_id"], "hybrid_score": original["hybrid_score"],
              "vacancy_fields": {key: original[key] for key in ("title", "requirements", "experience")},
              "requirements": [{"requirement": "SAP", "score": 3}],
              "overall_score": 3, "overall_explanation": "SAP expert"}
    assert score(row=nested)["baseline_score"] == score(row=original)["baseline_score"]


@pytest.mark.parametrize("language_text", ["English B2 optional.", "English B2 preferred.",
                                          "Желательно English B2.", "English B2 желательно."])
def test_an_optional_language_level_is_not_a_hard_requirement_or_bonus(language_text):
    cv = "Backend developer. Built services with Python. English A2."
    plain = score(cv, vacancy(requirements="Python.", experience=""))
    optional = score(cv, vacancy(requirements="Python. " + language_text, experience=""))
    assert components(optional)["languages"]["contribution"] == 0
    assert optional["baseline_score"] == plain["baseline_score"]
    assert not any(item["status"] == "mismatch"
                   for item in optional["baseline_evidence"]["languages"]["requirements"])


def test_actual_ml1_result_contract_is_preserved_with_a_controlled_index(tmp_path, monkeypatch, capsys):
    """Exercise JobMatcher's real adapter, without downloading an encoder/model."""
    from ML1 import matcher as matcher_module
    from ML1.config import EMBEDDING_DIMENSION
    from ML1.vacancy_index import read_vacancies, save_index

    source = tmp_path / "jobs.csv"
    fields = ["vacancy_id", "text", "title", "requirements", "responsibilities",
              "experience", "skills", "company", "city", "url"]
    rows = [
        {"vacancy_id": "controlled-backend", "text": "Python FastAPI PostgreSQL backend services",
         "title": "Backend Developer", "requirements": "Python, FastAPI, PostgreSQL",
         "responsibilities": "Build backend services", "experience": "От 2 лет",
         "skills": "python, fastapi, postgresql", "company": "Fictional API team",
         "city": "Алматы", "url": "https://example.invalid/backend"},
        {"vacancy_id": "controlled-data", "text": "SAP Tableau ETL data pipelines",
         "title": "Data Engineer", "requirements": "SAP, Tableau",
         "responsibilities": "Build ETL data pipelines", "experience": "От 3 лет",
         "skills": "sap, tableau", "company": "Fictional Data team",
         "city": "Астана", "url": "https://example.invalid/data"},
    ]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    frame, checksum = read_vacancies(source)
    matrix_path = tmp_path / "controlled_vectors.npy"
    save_index(frame, checksum, np.eye(2, EMBEDDING_DIMENSION, dtype=float), matrix_path)
    monkeypatch.setattr(matcher_module, "VACANCIES_FILE", source)
    monkeypatch.setattr(matcher_module, "EMBEDDINGS_FILE", matrix_path)
    vector = np.zeros(EMBEDDING_DIMENSION)
    vector[0], vector[1] = .8, .6
    calls = []

    def controlled_encoder(text):
        calls.append(text)
        return vector

    monkeypatch.setattr(matcher_module, "encode_text", controlled_encoder)
    initial = matcher_module.JobMatcher().match(CV, top_k=2)
    before = copy.deepcopy(initial)
    ranked = TransparentBaseline().rerank(CV, initial, top_k=2)
    capsys.readouterr()
    assert calls == [CV]
    assert initial == before
    assert {row["vacancy_id"] for row in ranked} == {"controlled-backend", "controlled-data"}
    by_id = {row["vacancy_id"]: (index, row) for index, row in enumerate(before, 1)}
    for row in ranked:
        position, original = by_id[row["vacancy_id"]]
        assert row["ml1_rank"] == position
        for key, value in original.items():
            assert row[key] == value
        assert len(row["matching_metadata"]["index_version"]) == 64


def test_cli_demo_returns_a_provenance_labelled_json_result():
    run = cli("--demo")
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["results"]
    assert all("baseline_score" in row and "hybrid_score" in row for row in payload["results"])
    assert any(row.get("matching_metadata", {}).get("fixture") for row in payload["results"])
    assert "ml1_initial_order" in payload
    assert "score_note" in payload


def test_cli_accepts_csv_ml1_scores_and_refuses_overwriting_a_result(tmp_path):
    cv = tmp_path / "cv.txt"
    cv.write_text(CV, encoding="utf-8")
    rows = [vacancy("z"), vacancy("a", hybrid_score=.2)]
    source = tmp_path / "shortlist.csv"
    fields = ["vacancy_id", "title", "requirements", "experience", "hybrid_score", "skill_score", "semantic_score"]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    output = tmp_path / "ranked.json"
    run = cli("--cv", cv, "--vacancies", source, "--output", output)
    assert run.returncode == 0, run.stderr
    payload = json.loads(output.read_text())
    assert [row["vacancy_id"] for row in payload["results"]] == ["z", "a"]
    before = output.read_bytes()
    repeated = cli("--cv", cv, "--vacancies", source, "--output", output)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**{key: rows[0][key] for key in fields}, "hybrid_score": "unknown"})
    invalid = cli("--cv", cv, "--vacancies", source)
    assert invalid.returncode != 0


def test_cli_json_and_human_summary_use_the_same_input_order_for_ties(tmp_path):
    cv = tmp_path / "cv.txt"
    cv.write_text(CV, encoding="utf-8")
    source = tmp_path / "shortlist.json"
    source.write_text(json.dumps([vacancy("z-job"), vacancy("a-job")]), encoding="utf-8")
    run = cli("--cv", cv, "--vacancies", source, "--top-k", "1")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["results"][0]["vacancy_id"] == "z-job"
    brief = cli("--cv", cv, "--vacancies", source, "--brief")
    assert brief.returncode == 0, brief.stderr
    assert brief.stdout.index("z-job") < brief.stdout.index("a-job")
