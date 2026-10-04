import json
import re
import uuid
from pathlib import Path

import pandas as pd

from NLP.skills_dictionary import SKILL_ALIASES

from NLP.annotation.annotation_dictionary import (
    ANNOTATION_HARD_SKILLS,
    SOFT_SKILL_ALIASES,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

RESUMES_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "resumes.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "resume_preannotated_tasks.json"
)


# ==========================================================
# Helpers
# ==========================================================

def create_region(start, end, text, label):
    """
    Create one Label Studio prediction region.
    """

    return {
        "id": str(uuid.uuid4())[:8],
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": text[start:end],
            "labels": [label],
        },
    }


def overlaps(start, end, existing_regions):
    """
    Prevent overlapping automatic annotations.
    """

    for region in existing_regions:

        value = region["value"]

        existing_start = value["start"]
        existing_end = value["end"]

        if (
            start < existing_end
            and end > existing_start
        ):
            return True

    return False


def add_region(
    regions,
    start,
    end,
    text,
    label
):
    """
    Add region if valid and not overlapping.
    """

    if start >= end:
        return

    if overlaps(
        start,
        end,
        regions
    ):
        return

    regions.append(
        create_region(
            start,
            end,
            text,
            label
        )
    )


# ==========================================================
# HARD SKILLS
# ==========================================================

def find_aliases(text, dictionary, label):
    """
    Find dictionary aliases and return Label Studio
    candidate spans.

    Longer aliases are preferred so that, for example,
    'Spring Security' is preferred over 'Spring'.
    """

    candidates = []

    variants = []

    for canonical, aliases in dictionary.items():

        all_variants = set(
            [canonical] + aliases
        )

        for variant in all_variants:

            variants.append(
                (
                    variant,
                    label
                )
            )

    # Longer strings first
    variants.sort(
        key=lambda item: len(item[0]),
        reverse=True
    )

    occupied = []

    for variant, entity_label in variants:

        pattern = (
            rf"(?<!\w)"
            rf"{re.escape(variant)}"
            rf"(?!\w)"
        )

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            start = match.start()
            end = match.end()

            collision = any(
                start < existing_end
                and end > existing_start
                for existing_start, existing_end
                in occupied
            )

            if collision:
                continue

            candidates.append(
                (
                    start,
                    end,
                    entity_label
                )
            )

            occupied.append(
                (
                    start,
                    end
                )
            )

    return candidates


def find_hard_skills(text):
    """
    HARD_SKILL candidates from both the original
    matching dictionary and pilot-derived additions.
    """

    combined_dictionary = {
        **SKILL_ALIASES,
        **ANNOTATION_HARD_SKILLS,
    }

    return find_aliases(
        text,
        combined_dictionary,
        "HARD_SKILL"
    )


def find_soft_skills(text):
    """
    SOFT_SKILL candidates discovered from the pilot.
    """

    return find_aliases(
        text,
        SOFT_SKILL_ALIASES,
        "SOFT_SKILL"
    )


# ==========================================================
# LANGUAGES
# ==========================================================

LANGUAGE_NAMES = [
    "English",
    "Russian",
    "Kazakh",
    "German",
    "French",
    "Chinese",
    "Spanish",
    "Turkish",
    "Korean",
    "Japanese",

    "Английский",
    "Русский",
    "Казахский",
    "Немецкий",
    "Французский",
    "Китайский",

    "Ағылшын",
    "Орыс",
    "Қазақ",
]


LANG_LEVEL = (
    r"(?:"
    r"A1|A2|B1|B2|C1|C2|"
    r"B2-C1|B1-B2|"
    r"Native|Fluent|Intermediate|Advanced|"
    r"Upper[- ]Intermediate|"
    r"Pre[- ]Intermediate|"
    r"Beginner|Elementary|"
    r"native speaker|"
    r"родной|свободно|свободный|"
    r"разговорный|"
    r"базовый"
    r")"
)


def find_languages(text):
    """
    Detect language + optional proficiency as one LANG span.
    """

    candidates = []

    language_pattern = "|".join(
        re.escape(language)
        for language in LANGUAGE_NAMES
    )

    pattern = (
        rf"\b(?:{language_pattern})\b"
        rf"(?:"
        rf"\s*"
        rf"(?:—|-|:|\(|/)?"
        rf"\s*"
        rf"{LANG_LEVEL}"
        rf"\)?"
        rf")?"
    )

    for match in re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE
    ):

        candidates.append(
            (
                match.start(),
                match.end(),
                "LANG"
            )
        )

    return candidates


# ==========================================================
# EDUCATION
# ==========================================================

EDUCATION_PATTERNS = [

    r"\bBachelor(?:'s)? Degree in [A-Za-z][A-Za-z &+\-/]+",

    r"\bBachelor(?:'s)? of [A-Za-z][A-Za-z &+\-/]+",

    r"\bBachelor(?:'s)? student in [A-Za-z][A-Za-z &+\-/]+",

    r"\bBachelor(?:'s)? Degree\b",

    r"\bBachelor(?:'s)? student\b",

    r"\bB\.?S\.? in [A-Za-z][A-Za-z &+\-/]+",

    r"\bB\.?Sc\.? in [A-Za-z][A-Za-z &+\-/]+",

    r"\bMaster(?:'s)? Degree in [A-Za-z][A-Za-z &+\-/]+",

    r"\bMaster(?:'s)? of [A-Za-z][A-Za-z &+\-/]+",

    r"\bM\.?S\.? in [A-Za-z][A-Za-z &+\-/]+",

    r"\bPhD in [A-Za-z][A-Za-z &+\-/]+",

    r"\bБакалавриат[,:\s]+[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ &+\-/]+",

    r"\bмагистратура[,:\s]+[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ &+\-/]+",
]


def find_education(text):

    candidates = []

    for pattern in EDUCATION_PATTERNS:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            candidates.append(
                (
                    match.start(),
                    match.end(),
                    "EDUCATION"
                )
            )

    return candidates


# ==========================================================
# EXPERIENCE
# ==========================================================

ROLE_WORDS = (
    r"(?:"
    r"Developer|Engineer|Analyst|Scientist|"
    r"Specialist|Intern|Manager|Tutor|"
    r"Administrator|Designer|Researcher|"
    r"Consultant|Lead|Founder|Co-founder|"
    r"Volunteer|Cashier"
    r")"
)


EXPERIENCE_PATTERNS = [

    # Internship roles
    r"\b[A-Za-z0-9+/#.-]+(?:\s+[A-Za-z0-9+/#.-]+){0,3}\s+Intern\b",

    # Common explicit job roles
    r"\bData Quality Analyst\b",
    r"\bData Analyst\b",
    r"\bData Scientist\b",
    r"\bML/AI Engineer Intern\b",
    r"\bProject Manager Assistant\b",
    r"\bPrivate Tutor\b",
    r"\bIT Specialist\b",
    r"\bMath Data Specialist\b",

    # Founder / lead roles
    r"\bCo-founder\s*&\s*Lead Developer\b",

    # Explicit experience durations
    r"\b\d+\+?\s+years?\s+of\s+experience\b",

    r"\b\d+\+?\s+years?\s+experience\b",

    r"\b\d+\+?\s+лет\s+опыта\b",

    r"\bопыт\s+работы\s+(?:от\s+)?\d+\+?\s+лет\b",
]


def find_experience(text):

    candidates = []

    for pattern in EXPERIENCE_PATTERNS:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            candidates.append(
                (
                    match.start(),
                    match.end(),
                    "EXPERIENCE"
                )
            )

    return candidates


# ==========================================================
# Process one resume
# ==========================================================

def generate_annotations(text):

    regions = []

    candidate_groups = [
        find_languages(text),
        find_education(text),
        find_experience(text),
        find_hard_skills(text),
        find_soft_skills(text),
    ]

    for candidates in candidate_groups:

        for start, end, label in candidates:

            add_region(
                regions,
                start,
                end,
                text,
                label
            )

    # Sort by text position
    regions.sort(
        key=lambda region:
        region["value"]["start"]
    )

    return regions


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("AUTO-HR RESUME PRE-ANNOTATION")
    print("=" * 70)

    df = pd.read_csv(
        RESUMES_FILE
    )

    # Ignore unreadable CVs
    df = df[
        df["character_count"] > 0
    ].copy()

    print(
        f"\nReadable resumes: {len(df)}"
    )

    tasks = []

    total_predictions = 0

    for _, row in df.iterrows():

        text = str(
            row["text"]
        )

        resume_id = row[
            "resume_id"
        ]

        regions = generate_annotations(
            text
        )

        total_predictions += len(
            regions
        )

        task = {
            "data": {
                "text": text,
                "resume_id": resume_id,
            },

            "predictions": [
                {
                    "model_version":
                        "autohr-rules-v2",

                    "score":
                        0.6,

                    "result":
                        regions,
                }
            ],
        }

        tasks.append(task)

        print(
            f"{resume_id}: "
            f"{len(regions)} predictions"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tasks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("PRE-ANNOTATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResumes: "
        f"{len(tasks)}"
    )

    print(
        f"Automatic regions: "
        f"{total_predictions}"
    )

    if tasks:

        print(
            f"Average regions/CV: "
            f"{total_predictions / len(tasks):.2f}"
        )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
