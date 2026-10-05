import os

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"

import sys
from pathlib import Path

from ML1.job_assistant import JobAssistant
from ML1.matcher import JobMatcher
from NLP.cv_parser import parse_cv


def print_result(position, result):
    """
    Display one vacancy result.
    """

    print("\n" + "=" * 70)
    print(f"#{position} {result['title']}")
    print("=" * 70)

    print(f"City: {result['city']}")
    print(f"Category: {result['job_category']}")

    print()

    print(
        f"Hybrid score:   "
        f"{result['hybrid_score']:.2%}"
    )

    print(
        f"Skill score:    "
        f"{result['skill_score']:.2%}"
    )

    print(
        f"Semantic score: "
        f"{result['semantic_score']:.2%}"
    )

    print()

    vacancy_skills = result.get(
        "vacancy_skills",
        ""
    )

    print(
        "Vacancy skills:",
        vacancy_skills
        if vacancy_skills
        else "None detected"
    )

    matched = result.get(
        "matched_skills",
        []
    )

    missing = result.get(
        "missing_skills",
        []
    )

    print(
        "Matched skills:",
        ", ".join(matched)
        if matched
        else "None"
    )

    print(
        "Missing skills:",
        ", ".join(missing)
        if missing
        else "None"
    )


def select_vacancy(results):
    """
    Let the candidate select one vacancy
    from the Top-K recommendations.
    """

    print("\n" + "=" * 70)
    print("SELECT VACANCY")
    print("=" * 70)

    print(
        f"\nEnter a vacancy number from 1 to "
        f"{len(results)}."
    )

    print(
        "Press Enter to exit."
    )

    while True:

        choice = input(
            "\nVacancy number: "
        ).strip()

        if not choice:
            return None

        try:
            position = int(choice)

        except ValueError:

            print(
                "Please enter a valid number."
            )

            continue

        if (
            position < 1
            or position > len(results)
        ):

            print(
                f"Please choose a number "
                f"between 1 and {len(results)}."
            )

            continue

        return results[
            position - 1
        ]


def main():

    print("=" * 70)
    print("AUTO-HR CV MATCHING SYSTEM")
    print("=" * 70)

    # --------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------

    if len(sys.argv) != 2:

        print("\nUsage:")

        print(
            'python -m ML1.match_cv '
            '"data/resumes/raw/example.pdf"'
        )

        sys.exit(1)

    cv_path = Path(
        sys.argv[1]
    )

    if not cv_path.exists():

        print(
            f"\nCV file not found:\n"
            f"{cv_path}"
        )

        sys.exit(1)

    # --------------------------------------------------
    # Parse CV
    # --------------------------------------------------

    print("\nReading CV...")

    try:

        cv = parse_cv(
            cv_path
        )

    except Exception as error:

        print(
            f"\nCould not process CV:\n"
            f"{error}"
        )

        sys.exit(1)

    resume_text = cv[
        "text"
    ]

    resume_skills = cv[
        "skills"
    ]

    if not resume_text.strip():

        print(
            "\nNo text could be extracted "
            "from this CV."
        )

        print(
            "This PDF may require OCR."
        )

        sys.exit(1)

    print(
        "CV successfully parsed."
    )

    print(
        f"Characters extracted: "
        f"{len(resume_text)}"
    )

    print(
        f"Skills detected: "
        f"{len(resume_skills)}"
    )

    print("\nDetected skills:")

    if resume_skills:

        for skill in resume_skills:
            print(
                f"  - {skill}"
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------
    # Load matcher
    # --------------------------------------------------

    print(
        "\nLoading job matcher..."
    )

    matcher = JobMatcher(
        alpha=0.4
    )

    # --------------------------------------------------
    # Match CV
    # --------------------------------------------------

    print(
        "\nComparing CV against vacancies..."
    )

    results = matcher.match(
        resume_text=resume_text,
        resume_skills=resume_skills,
        top_k=10
    )

    # --------------------------------------------------
    # Display Top 10
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP 10 MATCHING VACANCIES")
    print("=" * 70)

    for position, result in enumerate(
        results,
        start=1
    ):

        print_result(
            position,
            result
        )

    # --------------------------------------------------
    # Select vacancy
    # --------------------------------------------------

    selected_vacancy = (
        select_vacancy(
            results
        )
    )

    if selected_vacancy is None:

        print(
            "\nAUTO-HR closed."
        )

        return

    print(
        f"\nSelected vacancy: "
        f"{selected_vacancy['title']}"
    )

    print(
        "Starting AUTO-HR Assistant..."
    )

    # --------------------------------------------------
    # Start contextual assistant
    # --------------------------------------------------

    assistant = JobAssistant(
        resume_text=resume_text,
        resume_skills=resume_skills,
        vacancy=selected_vacancy
    )

    assistant.chat()


if __name__ == "__main__":
    main()