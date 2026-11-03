"""Grounded helper behavior on synthetic professional source, without an API."""
from copy import deepcopy
import math

import pytest

from backend.advice import answer_question, build_resume_feedback
from backend.matching import Matching, refresh_skills


@pytest.fixture(scope="module")
def engine():
    return Matching()


@pytest.fixture
def vacancy():
    return {"id": "synthetic-vacancy", "title": "Middle Backend Developer",
            "company": "Example IT Team", "city": "Алматы", "experience": "От 3 лет",
            "requirements": "Python, FastAPI. Optional: React.",
            "description": "Build backend services with Python and FastAPI.",
            "responsibilities": "", "skills": ["python", "fastapi", "react"],
            "salary": "", "employment": "Полная занятость", "is_custom": True,
            "url": "", "publish_date": "2026-11-03"}


def candidate(text):
    return refresh_skills({"id": "synthetic-candidate", "name": "Пример кандидата", "filename": "",
                           "text": text, "notes": "", "is_demo": False, "created_at": "2026-11-03",
                           "clarifications": [], "manual_added": [], "manual_removed": [],
                           "manual_confirmed": [], "vacancy_ids": []})


def test_general_resume_feedback_has_no_invented_vacancy_score():
    record = candidate("Backend developer. Python, FastAPI, PostgreSQL.")
    feedback = build_resume_feedback(record)
    assert "не рассчитывалось" in feedback["summary"]
    assert "match" not in feedback
    assert any("проект" in item for item in feedback["recommendations"])
    assert not any("вероятность найма" in item for item in feedback["strengths"])


def test_feedback_separates_needed_skills_from_optional_wishes(engine, vacancy):
    record = candidate("Middle Backend developer. Work experience: 3 years. Built Python services with FastAPI.")
    match = engine.pair(record, vacancy)
    assert "react" in match["missing_skills"]
    result = build_resume_feedback(record, vacancy, match)
    assert "2 из 2" in result["summary"]
    assert not any("react" in item.casefold() for item in result["gaps"] + result["recommendations"])
    assert result["match"] == match and result["match"] is not match


def test_feedback_shows_unknown_profile_duration_without_calling_it_zero(engine, vacancy):
    record = candidate("Middle Backend developer. Built Python services with FastAPI.")
    result = build_resume_feedback(record, vacancy, engine.pair(record, vacancy))
    assert any("не установлена" in item for item in result["gaps"])
    assert not any("0 г." in item for item in result["gaps"])


def test_score_question_without_resume_requests_selection(vacancy):
    result = answer_question("Почему такой низкий score?", vacancy)
    assert result["topic"] == "score"
    assert "Выберите своё резюме" in result["answer"]
    assert result["evidence"] == []
    assert "score_breakdown" not in result


def test_score_answer_uses_current_model_and_exact_rule(engine, vacancy):
    record = candidate("Middle Backend developer. Work experience: 1 year. Built Python services with FastAPI.")
    match = engine.pair(record, vacancy)
    result = answer_question("За что мне дали этот балл?", vacancy, record, match)
    assert result["topic"] == "score"
    assert "не процент пригодности" in result["answer"]
    assert "Недостаточный профильный стаж" in result["answer"]
    assert result["score_breakdown"] == match["score_breakdown"]
    calculation = next(item["quote"] for item in result["evidence"] if item["source"] == "calculation")
    assert match["score_breakdown"]["model_version"] in calculation
    assert "сумма поправок" in calculation
    details = result["score_breakdown"]
    assert math.isclose(details["raw_score"] + sum(item["contribution"] for item in details["adjustments"]), details["final_score"], abs_tol=1e-12)


def test_salary_is_never_inferred_from_title_or_score(vacancy):
    unknown = answer_question("Какая зарплата?", vacancy)
    assert "зарплата не указана" in unknown["answer"]
    assert unknown["evidence"] == []
    specified = answer_question("What salary is offered?", {**vacancy, "salary": "500 000–800 000 ₸"})
    assert "500 000–800 000 ₸" in specified["answer"]
    assert specified["evidence"] == [{"source": "vacancy", "quote": "500 000–800 000 ₸"}]


def test_questions_do_not_create_company_culture_or_current_job_status(vacancy):
    result = answer_question("Какая культура в компании и дружный ли коллектив?", vacancy)
    assert result["topic"] == "unknown"
    assert "нет достаточного ответа" in result["answer"]
    assert result["evidence"] == []
    remote = answer_question("Есть ли удалённая работа?", vacancy)
    assert "не подтверждено" in remote["answer"]
    assert "Полная занятость" not in remote["answer"]
    assert remote["evidence"] == []


@pytest.mark.parametrize("source", [
    "Частичная занятость, 20 часов в неделю.",
    "Полная занятость, 40 часов в неделю.",
    "Part-time, 20 hours per week.",
    "Full-time, 40 hours per week.",
])
def test_employment_answer_uses_explicit_source_instead_of_inventing_load(vacancy, source):
    result = answer_question("Какая занятость у вакансии?", {**vacancy, "employment": "", "description": source})
    assert result["topic"] == "employment"
    assert source in result["answer"]
    assert result["evidence"] == [{"source": "vacancy", "quote": source}]
    assert "удалённ" not in result["answer"]


def test_remote_and_working_hours_are_separate_from_full_time(vacancy):
    schedule = answer_question("Какой рабочий график?", vacancy)
    assert "не уточнены" in schedule["answer"]
    assert "Полная занятость" not in schedule["answer"]
    assert schedule["evidence"] == []
    source = "Remote work is not available. Part-time, 20 hours per week."
    specified = {**vacancy, "employment": "", "description": source}
    remote = answer_question("Can I work remote?", specified)
    assert "Remote work is not available." in remote["answer"]
    assert "Part-time" not in remote["answer"]
    assert remote["evidence"] == [{"source": "vacancy", "quote": "Remote work is not available."}]
    schedule = answer_question("What is the work schedule?", specified)
    assert schedule["evidence"] == [{"source": "vacancy", "quote": "Part-time, 20 hours per week."}]


def test_specific_skill_answer_distinguishes_optional_and_denied(engine, vacancy):
    record = candidate("Backend developer. I used Python and have not used FastAPI. React.")
    match = engine.pair(record, vacancy)
    result = answer_question("Нужны FastAPI и React?", vacancy, record, match)
    assert "fastapi: найден" in result["answer"]
    assert "react: упомянут" in result["answer"]
    assert "недостаточно подтверждений: fastapi" in result["answer"]
    assert all(item["quote"] in "\n".join(str(value) for value in vacancy.values()) for item in result["evidence"])


def test_experience_answer_keeps_general_and_relevant_years_separate(engine, vacancy):
    record = candidate("Senior frontend developer — 5 years. Junior backend developer — 1 year. Work experience: 6 years. Built Python services with FastAPI.")
    result = answer_question("Сколько опыта у меня для этой работы?", vacancy, record, engine.pair(record, vacancy))
    assert "Общий заявленный стаж: 6 г.; по этой роли: 1 г." in result["answer"]
    assert "От 3 лет" in result["answer"]
    assert any(item["source"] == "resume" for item in result["evidence"])


def test_helper_does_not_mutate_candidate_or_matching_and_preserves_answer_source(engine, vacancy):
    record = candidate("Middle Backend developer. Work experience: 3 years. Built backend services with Python.")
    record["clarifications"].append({"question": "Использовали FastAPI?", "answer": "Built FastAPI services in my backend project."})
    refresh_skills(record)
    match = engine.pair(record, vacancy)
    before = deepcopy((record, vacancy, match))
    feedback = build_resume_feedback(record, vacancy, match)
    response = answer_question("Почему такой score?", vacancy, record, match)
    assert (record, vacancy, match) == before
    assert any(item["source"] == "clarification" for item in response["evidence"])
    response["score_breakdown"]["contributions"].clear()
    feedback["match"]["candidate"]["name"] = "Changed output only"
    assert (record, vacancy, match) == before


def test_unknown_question_and_blank_are_explicit(vacancy):
    assert answer_question("Кто победит в футболе?", vacancy)["topic"] == "unknown"
    with pytest.raises(ValueError):
        answer_question("  ", vacancy)


def test_overview_reports_source_without_promising_job_is_still_open(vacancy):
    result = answer_question("Расскажи об этой вакансии", vacancy)
    assert result["topic"] == "overview"
    assert vacancy["title"] in result["answer"]
    assert vacancy["company"] in result["answer"]
    assert "актуальность" in result["answer"]
    assert all(proof["source"] == "vacancy" for proof in result["evidence"])


def test_language_and_education_answers_quote_requirements_or_state_unknown(vacancy):
    specified = {**vacancy, "requirements": "Python and FastAPI. English B2 required. Bachelor degree in Computer Science."}
    language = answer_question("Какой английский требуется?", specified)
    assert language["topic"] == "languages"
    assert "English B2 required." in language["answer"]
    assert language["evidence"] == [{"source": "vacancy", "quote": "English B2 required."}]
    education = answer_question("Нужен ли диплом?", specified)
    assert education["topic"] == "education"
    assert "Bachelor degree in Computer Science." in education["answer"]
    unknown = answer_question("Какое нужно образование?", vacancy)
    assert "не найдено" in unknown["answer"]
    assert unknown["evidence"] == []
