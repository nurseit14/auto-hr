"""Behavioral coverage for role-aware ranking and its additive explanations."""

import math

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(database_path=tmp_path / "breakdown.sqlite3")


@pytest.fixture
def client(app):
    with TestClient(app) as connection:
        yield connection


def candidate(client, text, vacancy_id="11", name="Проверка профессионального опыта"):
    response = client.post("/api/candidates", json={"name": name, "text": text, "vacancy_id": vacancy_id})
    assert response.status_code == 201, response.text
    return response.json()


def shortlist(client, vacancy_id="11"):
    response = client.post("/api/shortlist", json={"vacancy_id": vacancy_id})
    assert response.status_code == 200, response.text
    return response.json()["candidates"]


def assert_additive(row):
    breakdown = row["score_breakdown"]
    assert breakdown["model_version"] == "ml2-pairwise-linear-v2"
    assert breakdown["feature_version"] == "professional-signals-v2"
    assert len(breakdown["contributions"]) == 12
    assert math.isclose(breakdown["base_score"] + sum(item["contribution"] for item in breakdown["contributions"]),
                        breakdown["raw_score"], abs_tol=1e-10)
    assert math.isclose(breakdown["raw_score"] + sum(item["contribution"] for item in breakdown["adjustments"]),
                        breakdown["final_score"], abs_tol=1e-10)
    assert math.isclose(row["score"], breakdown["final_score"], abs_tol=5.1e-7)
    for item in breakdown["contributions"]:
        assert item["label"] and item["reason"]
        assert all(math.isfinite(item[key]) for key in ("value", "mean", "scale", "weight", "contribution"))
        assert math.isclose((item["value"] - item["mean"]) / item["scale"] * item["weight"],
                            item["contribution"], abs_tol=1e-10)
        assert all(evidence["source"] in {"resume", "clarification", "hr", "vacancy"}
                   and evidence["quote"] for evidence in item["evidence"])
    return breakdown


def test_sap_tableau_does_not_reward_unrelated_demo_years_or_middle(client):
    demo = client.post("/api/demo").json()
    for item in demo["candidates"]:
        response = client.put(f"/api/candidates/{item['id']}/vacancies",
                              json={"vacancy_ids": [demo["vacancy_id"], "11"]})
        assert response.status_code == 200
    rows = shortlist(client)
    assert len(rows) == 4
    assert all(row["matched_skills"] == [] and row["skill_score"] == 0 for row in rows)
    assert all(row["score"] <= 0 for row in rows)
    for row in rows:
        breakdown = assert_additive(row)
        signals = {item["feature"]: item for item in breakdown["contributions"]}
        assert signals["experience_compatibility"]["contribution"] <= 0
        assert signals["seniority_compatibility"]["contribution"] <= 0
        assert breakdown["experience"]["relevant_years"] is None
        assert breakdown["experience"]["required_years"] == 3
        assert breakdown["experience"]["reason"]


def test_actual_data_engineering_evidence_improves_rank_and_is_explained(client):
    unrelated = candidate(client, "Middle frontend developer. Work experience: 8 years. React TypeScript HTML CSS. Built web interfaces.")
    relevant = candidate(client, "Middle Data Engineer. Work experience: 4 years. Built ETL data pipelines and data warehouses. Worked with SAP, Tableau, SQL and Python.")
    rows = {row["candidate"]["id"]: row for row in shortlist(client)}
    assert rows[relevant["id"]]["score"] > rows[unrelated["id"]]["score"]
    assert set(rows[relevant["id"]]["matched_skills"]) == {"sap", "tableau"}
    breakdown = assert_additive(rows[relevant["id"]])
    assert breakdown["experience"]["relevant_years"] == 4
    assert breakdown["experience"]["general_years"] == 4
    assert breakdown["experience"]["evidence"]
    skill_items = [item for item in breakdown["contributions"] if item["feature"] in {"skill_coverage", "skill_jaccard"}]
    assert sum(item["contribution"] for item in skill_items) > 0
    assert any(item["evidence"] for item in skill_items)


def test_mixed_career_does_not_transfer_frontend_years_or_level_to_backend(client):
    response = client.post("/api/vacancies", json={
        "title": "Middle Backend Developer", "company": "Проверка стажа",
        "requirements": "Python, FastAPI, PostgreSQL, Docker. Build backend services.",
        "experience": "От 3 лет", "city": "Алматы",
    })
    assert response.status_code == 201, response.text
    vacancy_id = response.json()["id"]
    candidate(client, "Senior frontend developer: 5 years. Junior backend developer: 1 year. "
              "Work experience: 6 years. Built backend services with Python, FastAPI and PostgreSQL.", vacancy_id)
    breakdown = assert_additive(shortlist(client, vacancy_id)[0])
    assert breakdown["experience"]["general_years"] == 6
    assert breakdown["experience"]["relevant_years"] == 1
    assert breakdown["experience"]["required_years"] == 3
    signals = {item["feature"]: item for item in breakdown["contributions"]}
    assert signals["experience_compatibility"]["value"] < 0
    assert signals["seniority_compatibility"]["value"] < 0


def test_known_relevant_shortfall_has_a_separate_exact_explained_adjustment(client):
    response = client.post("/api/vacancies", json={
        "title": "Middle Data Engineer", "company": "Проверка профильного минимума",
        "requirements": "SAP and Tableau. Build ETL data pipelines.",
        "experience": "От 3 лет", "city": "Алматы",
    })
    assert response.status_code == 201, response.text
    vacancy_id = response.json()["id"]
    short = candidate(client, "Middle Data Engineer. Work experience: 1 year. "
                      "Built ETL data pipelines with SAP and Tableau.", vacancy_id, "Профильный год")
    enough = candidate(client, "Middle Data Engineer. Work experience: 3 years. "
                       "Built ETL data pipelines with SAP and Tableau.", vacancy_id, "Профильные три года")
    unknown = candidate(client, "Middle Data Engineer. Built ETL data pipelines with SAP and Tableau.",
                        vacancy_id, "Стаж не указан")
    rows = {row["candidate"]["id"]: row for row in shortlist(client, vacancy_id)}
    insufficient = assert_additive(rows[short["id"]])
    sufficient = assert_additive(rows[enough["id"]])
    unconfirmed = assert_additive(rows[unknown["id"]])
    assert insufficient["raw_score"] > 0
    assert insufficient["final_score"] < sufficient["final_score"]
    assert len(insufficient["adjustments"]) == 1
    rule = insufficient["adjustments"][0]
    assert rule["policy"] == "relevant-experience-shortfall-v1"
    assert math.isclose(rule["factor"], 1 / 3, abs_tol=1e-10)
    assert math.isclose(rule["contribution"], -insufficient["raw_score"] * 2 / 3, abs_tol=1e-10)
    assert math.isclose(insufficient["final_score"], insufficient["raw_score"] / 3, abs_tol=1e-10)
    assert "1 г." in rule["reason"] and "3 г." in rule["reason"]
    assert "не обученный коэффициент" in rule["reason"]
    assert {item["source"] for item in rule["evidence"]} == {"resume", "vacancy"}
    assert sufficient["adjustments"] == []
    assert unconfirmed["experience"]["relevant_years"] is None
    assert unconfirmed["adjustments"] == []
    assert unconfirmed["raw_score"] == unconfirmed["final_score"]


def test_answers_and_hr_skill_confirmation_retain_their_real_sources(client):
    saved = candidate(client, "Middle Data Engineer. Work experience: 4 years. Built production ETL data pipelines for a data warehouse.")
    before = shortlist(client)[0]
    response = client.post(f"/api/candidates/{saved['id']}/clarifications", json={
        "question": "Какой у вас опыт работы с SAP?",
        "answer": "Работал с SAP: настраивал выгрузку данных в хранилище и проверял преобразования.",
    })
    assert response.status_code in (200, 201), response.text
    answered = shortlist(client)[0]
    assert answered["score"] > before["score"]
    breakdown = assert_additive(answered)
    assert any(evidence["source"] == "clarification" and "SAP" in evidence["quote"]
               for item in breakdown["contributions"] for evidence in item["evidence"])
    response = client.patch(f"/api/candidates/{saved['id']}", json={"skills": ["sap", "tableau"]})
    assert response.status_code == 200, response.text
    corrected = shortlist(client)[0]
    assert set(corrected["matched_skills"]) == {"sap", "tableau"}
    assert corrected["score"] > answered["score"]
    breakdown = assert_additive(corrected)
    assert any(evidence["source"] == "hr" and "tableau" in evidence["quote"]
               for item in breakdown["contributions"] for evidence in item["evidence"])


def test_candidate_to_vacancies_exposes_the_same_additive_explanation(client):
    saved = candidate(client, "Middle Data Engineer. Work experience: 4 years. Built ETL pipelines with SAP, Tableau, Python and SQL.")
    response = client.post("/api/matches", json={"candidate_id": saved["id"], "limit": 3})
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 3
    for item in response.json()["items"]:
        assert_additive(item)


def test_learning_wishes_are_consistent_between_coverage_and_model(client):
    response = client.post("/api/vacancies", json={
        "title": "Middle Backend Developer", "company": "Проверка навыков",
        "requirements": "Python and FastAPI", "experience": "От 3 лет", "city": "Алматы",
    })
    assert response.status_code == 201, response.text
    vacancy_id = response.json()["id"]
    candidate(client, "Middle frontend developer. Work experience: 5 years. "
              "I want to learn Python and FastAPI. I have no experience with backend services.", vacancy_id)
    row = shortlist(client, vacancy_id)[0]
    assert row["matched_skills"] == []
    assert set(row["uncertain_skills"]) == {"python", "fastapi"}
    assert row["skill_score"] == 0
    breakdown = assert_additive(row)
    signals = {item["feature"]: item for item in breakdown["contributions"]}
    assert signals["skill_coverage"]["value"] == 0
    assert signals["skill_jaccard"]["value"] == 0
    assert signals["experience_compatibility"]["value"] == 0
    assert signals["seniority_compatibility"]["value"] == 0


def test_missing_model_explanation_is_an_explicit_skill_baseline(client, app):
    app.state.matching.model = None
    candidate(client, "Middle Data Engineer. Work experience: 4 years. Built ETL data pipelines with SAP.")
    row = shortlist(client)[0]
    breakdown = row["score_breakdown"]
    assert breakdown["model_version"] == "offline-baseline"
    assert breakdown["feature_version"] == "dictionary-skills"
    assert len(breakdown["contributions"]) == 1
    assert breakdown["warnings"]
    assert breakdown["adjustments"] == []
    assert row["score"] == row["skill_score"] == breakdown["final_score"]
    assert breakdown["contributions"][0]["contribution"] == row["score"]
