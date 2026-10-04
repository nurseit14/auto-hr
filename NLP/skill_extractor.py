import re

from NLP.skills_dictionary import SKILL_ALIASES


def normalize_text(text):
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_alias(text, alias):
    """
    Check whether an alias occurs in text.

    Boundaries prevent things such as 'java' being
    accidentally detected inside another word.
    """
    pattern = rf"(?<!\w){re.escape(alias.lower())}(?!\w)"
    return re.search(pattern, text) is not None


def extract_skills(text):
    """
    Extract normalized IT skills from text.

    Returns:
        list[str]
    """

    text = normalize_text(text)

    found_skills = []

    for canonical_skill, aliases in SKILL_ALIASES.items():

        # Also check the canonical name itself.
        variants = [canonical_skill] + aliases

        for alias in variants:
            if contains_alias(text, alias):
                found_skills.append(canonical_skill)
                break

    return sorted(set(found_skills))


if __name__ == "__main__":

    sample = """
    Backend developer with experience in Python,
    FastAPI, PostgreSQL, Docker and Git.
    Experience building REST APIs.
    """

    skills = extract_skills(sample)

    print("Extracted skills:")

    for skill in skills:
        print(f"- {skill}")
