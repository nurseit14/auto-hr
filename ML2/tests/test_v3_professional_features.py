"""Source-level contract for v3 requirements and grounded professional signals."""
import pytest

from ML2.features import FEATURE_NAMES, extract_features


BACKEND = {"title": "Middle Backend Developer", "requirements": "Python, FastAPI, PostgreSQL.",
           "experience": "От 3 до 6 лет"}
DATA = {"title": "Middle Data Engineer", "requirements": "SAP, Tableau.",
        "responsibilities": "Build ETL data pipelines.", "experience": "От 3 до 6 лет"}


def features(text, vacancy):
    vector, facts = extract_features(text, vacancy)
    assert len(vector) == len(FEATURE_NAMES) == 15
    assert set(facts["feature_reasons"]) == set(FEATURE_NAMES)
    assert set(facts["feature_evidence"]) == set(FEATURE_NAMES)
    return dict(zip(FEATURE_NAMES, vector)), facts


def test_optional_skills_do_not_dilute_required_skill_coverage():
    vacancy = {**BACKEND, "requirements": "Python, FastAPI. Optional React, Docker."}
    values, facts = features("Backend developer. Python FastAPI.", vacancy)
    assert values["required_skill_coverage"] == 1
    assert values["skill_coverage"] == .5
    assert facts["required_skills"] == ["fastapi", "python"]


def test_inline_wishes_keep_preceding_requirements():
    vacancy = {**BACKEND, "requirements": "Python и SQL, желательно Docker и React."}
    values, facts = features("Backend developer. Python SQL.", vacancy)
    assert values["required_skill_coverage"] == 1
    assert facts["required_skills"] == ["python", "sql"]


@pytest.mark.parametrize("requirements", [
    "Python required. Docker optional.",
    "Python required. Docker (optional).",
    "Python required, Docker preferred.",
])
def test_postfix_optional_does_not_become_a_requirement(requirements):
    values, facts = features("Backend developer. Python.", {**BACKEND, "requirements": requirements})
    assert facts["required_skills"] == ["python"]
    assert values["required_skill_coverage"] == 1


def test_optional_heading_remains_optional_until_required_heading():
    vacancy = {**BACKEND, "requirements": "Желательно:\nDocker\nReact\nТребования:\nPython"}
    values, facts = features("Backend developer. Python.", vacancy)
    assert facts["required_skills"] == ["python"]
    assert values["required_skill_coverage"] == 1


def test_described_work_has_distinct_evidence_from_skill_list():
    listed, facts = features("Backend developer. Python, FastAPI, PostgreSQL.", BACKEND)
    worked, work_facts = features("Backend developer. Built Python FastAPI services with PostgreSQL.", BACKEND)
    assert listed["required_skill_coverage"] == worked["required_skill_coverage"] == 1
    assert listed["demonstrated_skill_coverage"] == 0
    assert worked["demonstrated_skill_coverage"] == 1
    assert all("built" in row["quote"] for row in work_facts["feature_evidence"]["demonstrated_skill_coverage"])


def test_learning_or_denial_does_not_count_as_demonstrated_work():
    values, facts = features("Middle Backend developer. Work experience: 5 years. Never used Python or FastAPI. I want to learn PostgreSQL.", BACKEND)
    assert values["required_skill_coverage"] == 0
    assert values["demonstrated_skill_coverage"] == 0
    assert values["relevant_experience_fulfillment"] == 0
    assert facts["relevant_years"] is None


def test_fullstack_can_supply_supported_backend_experience():
    values, facts = features("Middle fullstack developer. Work experience: 3 years. Built Python FastAPI services with PostgreSQL.", BACKEND)
    assert values["role_compatibility"] == 1
    assert values["seniority_compatibility"] == 1
    assert facts["relevant_years"] == 3
    assert values["relevant_experience_fulfillment"] == 1


def test_generic_software_vacancy_accepts_specific_development_profile():
    vacancy = {**BACKEND, "title": "Software Developer"}
    values, facts = features("Backend developer. Work experience: 3 years. Python FastAPI PostgreSQL.", vacancy)
    assert values["role_compatibility"] == 1
    assert facts["relevant_years"] == 3


def test_specialist_tasks_can_confirm_profile_without_inventing_years():
    values, facts = features("Software engineer. Work experience: 6 years. Built ETL data pipelines with SAP and Tableau.", DATA)
    assert values["role_compatibility"] == 1
    assert facts["relevance_supported"]
    assert facts["relevant_years"] is None
    assert values["relevant_experience_fulfillment"] == 0


def test_fulfillment_is_continuous_only_for_known_profile_years():
    short, short_facts = features("Middle Backend developer — 1 year. Built Python FastAPI PostgreSQL services.", BACKEND)
    full, _ = features("Middle Backend developer — 3 years. Built Python FastAPI PostgreSQL services.", BACKEND)
    unknown, unknown_facts = features("Middle Backend developer. Built Python FastAPI PostgreSQL services.", BACKEND)
    assert short_facts["relevant_years"] == 1
    assert short["relevant_experience_fulfillment"] == pytest.approx(1 / 3)
    assert full["relevant_experience_fulfillment"] == 1
    assert unknown["relevant_experience_fulfillment"] == 0
    assert unknown_facts["relevant_years"] is None


def test_unrelated_senior_with_only_common_tools_stays_unsupported():
    values, facts = features("Senior frontend developer. Work experience: 12 years. React TypeScript Git REST API.", DATA)
    assert values["role_compatibility"] == 0
    assert values["experience_compatibility"] == 0
    assert values["relevant_experience_fulfillment"] == 0
    assert values["seniority_compatibility"] == 0
    assert facts["relevant_years"] is None


def test_mixed_career_does_not_borrow_earlier_frontend_years():
    values, facts = features("Senior frontend developer — 5 years. Junior backend developer — 1 year. Built Python FastAPI PostgreSQL services.", BACKEND)
    assert facts["relevant_years"] == 1
    assert values["relevant_experience_fulfillment"] == pytest.approx(1 / 3)
    assert values["seniority_compatibility"] < 0


@pytest.mark.parametrize("denial", [
    "Have not used Docker.",
    "I have never worked with Docker.",
    "Docker is a tool I have not used.",
    "Docker I have not used.",
    "I didn't use Docker.",
])
def test_entity_scoped_english_denial_is_not_skill_evidence(denial):
    values, facts = features("Middle backend developer. Work experience: 5 years. " + denial,
                             {**BACKEND, "requirements": "Docker"})
    assert facts["confirmed_skill_matches"] == []
    assert values["required_skill_coverage"] == 0
    assert values["demonstrated_skill_coverage"] == 0
    assert facts["relevant_years"] is None


def test_relative_clause_denial_does_not_confirm_tableau():
    values, facts = features("Middle Data Engineer. Work experience: 4 years. Tableau is a tool I have not used.", DATA)
    assert "tableau" not in facts["confirmed_skill_matches"]
    assert values["required_skill_coverage"] == 0
    assert values["demonstrated_skill_coverage"] == 0
    assert facts["relevant_years"] is None


@pytest.mark.parametrize("denial", [
    "Опыта построения потоков данных нет.",
    "Опыта построения ETL data pipelines нет.",
    "I have not built ETL data pipelines.",
])
def test_negative_task_clause_does_not_establish_professional_profile(denial):
    values, facts = features("Middle frontend developer. Work experience: 5 years. " + denial, DATA)
    assert values["task_coverage"] == 0
    assert values["role_compatibility"] == 0
    assert values["experience_compatibility"] == 0
    assert values["relevant_experience_fulfillment"] == 0
    assert facts["relevant_years"] is None


def test_mixed_positive_python_and_denied_docker_keep_distinct_evidence():
    values, facts = features("Backend developer. I used Python and have not used Docker.",
                             {**BACKEND, "requirements": "Python, Docker"})
    assert facts["confirmed_skill_matches"] == ["python"]
    assert values["required_skill_coverage"] == .5
    assert values["demonstrated_skill_coverage"] == .5
