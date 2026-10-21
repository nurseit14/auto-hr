"""Structured model answers and checks against the original documents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "ROLE_LEVEL", "EXPERIENCE", "HARD_SKILL", "SOFT_SKILL",
        "EDUCATION", "LANGUAGE", "OTHER",
    ]
    requirement: str
    vacancy_quote: str
    score: Literal[0, 1, 2, 3]
    status: Literal["unknown", "mismatch", "partial", "match"]
    cv_quote: str
    explanation: str


class PairAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[RequirementAssessment]
    overall_score: Literal[0, 1, 2, 3]
    overall_explanation: str
    warnings: list[str]


class ModelRequirementAssessment(BaseModel):
    """The model selects source line IDs; Python copies the actual CV quotes."""
    model_config = ConfigDict(extra="forbid")

    category: Literal["ROLE_LEVEL", "EXPERIENCE", "HARD_SKILL", "SOFT_SKILL", "EDUCATION", "LANGUAGE", "OTHER"]
    requirement: str
    vacancy_quote: str
    vacancy_evidence_line_id: int
    score: Literal[0, 1, 2, 3]
    status: Literal["unknown", "mismatch", "partial", "match"]
    cv_evidence_line_ids: list[int]
    explanation: str


class ModelPairAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirements: list[ModelRequirementAssessment]
    overall_score: Literal[0, 1, 2, 3]
    overall_explanation: str
    warnings: list[str]


def numbered_lines(text: str) -> list[dict]:
    lines, offset = [], 0
    for index, line in enumerate(text.splitlines(keepends=True), 1):
        if line.strip():
            lines.append({"line_id": index, "text": line.rstrip("\r\n"), "start": offset,
                          "end": offset + len(line.rstrip("\r\n"))})
        offset += len(line)
    return lines


def ground_model_answer(answer: ModelPairAssessment, cv_text: str, vacancy_display: str) -> PairAssessment:
    lines = {line["line_id"]: line for line in numbered_lines(cv_text)}
    vacancy_lines = {line["line_id"]: line for line in numbered_lines(vacancy_display)}
    rows = []
    for row in answer.requirements:
        ids = row.cv_evidence_line_ids
        if row.score == 0 and ids:
            raise ValueError(f"{row.requirement}: unknown must have empty cv_evidence_line_ids")
        if row.score > 0 and not ids:
            raise ValueError(f"{row.requirement}: scores 1–3 require source line IDs")
        if any(line_id not in lines for line_id in ids):
            raise ValueError(f"{row.requirement}: a cv_evidence_line_id does not exist")
        quote = ""
        if ids:
            # Copy a continuous original span, including intermediate context if needed.
            quote = cv_text[min(lines[i]["start"] for i in ids):max(lines[i]["end"] for i in ids)]
        if row.vacancy_evidence_line_id not in vacancy_lines:
            raise ValueError(f"{row.requirement}: vacancy_evidence_line_id does not exist")
        source_line = vacancy_lines[row.vacancy_evidence_line_id]["text"]
        values = row.model_dump(exclude={"cv_evidence_line_ids", "vacancy_evidence_line_id"})
        # The selected line is canonical. Keep a shorter highlight only if it is verbatim.
        if not row.vacancy_quote.strip() or row.vacancy_quote not in source_line:
            values["vacancy_quote"] = source_line
        rows.append(RequirementAssessment(**values, cv_quote=quote))
    return PairAssessment(requirements=rows, overall_score=answer.overall_score,
                          overall_explanation=answer.overall_explanation, warnings=answer.warnings)


def validate_assessment(answer: PairAssessment, cv_text: str, vacancy: dict) -> None:
    """Reject fabricated quotes, inconsistent scores and missing core categories."""
    source = vacancy["display"]
    if not answer.requirements:
        raise ValueError("No requirements returned")
    expected = {0: "unknown", 1: "mismatch", 2: "partial", 3: "match"}
    seen = set()
    categories = set()
    for row in answer.requirements:
        if not row.requirement.strip() or not row.explanation.strip():
            raise ValueError("Requirement and explanation must be nonempty")
        if not row.vacancy_quote.strip() or row.vacancy_quote not in source:
            raise ValueError("vacancy_quote must be an exact nonempty source substring")
        if row.cv_quote and row.cv_quote not in cv_text:
            raise ValueError("cv_quote must be an exact CV substring")
        if row.status != expected[row.score]:
            raise ValueError("score and status disagree")
        if row.score == 0 and row.cv_quote:
            raise ValueError("unknown must have an empty cv_quote")
        if row.score > 0 and not row.cv_quote.strip():
            raise ValueError("Scores 1–3 require direct CV evidence")
        if not cv_text.strip() and row.score != 0:
            raise ValueError("An empty CV can only receive unknown requirement scores")
        key = (row.category, row.requirement.casefold().strip())
        if key in seen:
            raise ValueError("Duplicate requirement")
        seen.add(key)
        categories.add(row.category)
    if vacancy.get("title") and "ROLE_LEVEL" not in categories:
        raise ValueError("The vacancy title requires a ROLE_LEVEL assessment")
    if vacancy.get("experience") and "EXPERIENCE" not in categories:
        raise ValueError("The structured experience field must be assessed")
    if not answer.overall_explanation.strip():
        raise ValueError("Overall explanation must be nonempty")
    if all(row.score == 0 for row in answer.requirements) and answer.overall_score != 0:
        raise ValueError("All-unknown requirements require an unknown overall assessment")
    if answer.overall_score == 0 and not any(row.score == 0 for row in answer.requirements):
        raise ValueError("Overall unknown needs missing evidence")
    if answer.overall_score == 3 and any(row.category == "ROLE_LEVEL" and row.score == 1 for row in answer.requirements):
        raise ValueError("A confirmed role/level mismatch cannot receive overall_score=3")
