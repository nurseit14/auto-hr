def parse_skills(skills):
    """
    Convert skills into a normalized set.

    Supports:
    - comma-separated string
    - list
    - set
    - None
    """

    if skills is None:
        return set()

    if isinstance(skills, str):
        return {
            skill.strip().lower()
            for skill in skills.split(",")
            if skill.strip()
        }

    if isinstance(skills, (list, set, tuple)):
        return {
            str(skill).strip().lower()
            for skill in skills
            if str(skill).strip()
        }

    return set()


def calculate_skill_similarity(
    resume_skills,
    vacancy_skills,
    confidence_target=5
):
    """
    Calculate confidence-adjusted skill similarity.

    The basic score measures how much of the detected
    vacancy skill set is covered by the resume.

    A confidence factor reduces the influence of vacancies
    where only a very small number of skills were detected.
    """

    resume_skills = parse_skills(resume_skills)
    vacancy_skills = parse_skills(vacancy_skills)

    if not vacancy_skills:
        return 0.0

    matched = resume_skills.intersection(
        vacancy_skills
    )

    # Basic coverage
    coverage = (
        len(matched)
        / len(vacancy_skills)
    )

    # Confidence in vacancy skill representation
    confidence = min(
        1.0,
        len(vacancy_skills)
        / confidence_target
    )

    adjusted_score = (
        coverage * confidence
    )

    return adjusted_score


def get_skill_details(
    resume_skills,
    vacancy_skills
):
    """
    Return matched and missing skills.
    """

    resume_skills = parse_skills(resume_skills)
    vacancy_skills = parse_skills(vacancy_skills)

    matched = sorted(
        resume_skills.intersection(vacancy_skills)
    )

    missing = sorted(
        vacancy_skills - resume_skills
    )

    return matched, missing


if __name__ == "__main__":

    resume = [
        "python",
        "docker",
        "postgresql",
        "git"
    ]

    vacancy = [
        "python",
        "docker",
        "postgresql",
        "kubernetes"
    ]

    score = calculate_skill_similarity(
        resume,
        vacancy
    )

    matched, missing = get_skill_details(
        resume,
        vacancy
    )

    print("Skill score:", score)
    print("Matched:", matched)
    print("Missing:", missing)
