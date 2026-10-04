import sys
from pathlib import Path

from ML1.matcher import JobMatcher
from NLP.cv_parser import parse_cv


def print_result(position, result):

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

    matched = result["matched_skills"]
    missing = result["missing_skills"]

    print(
        "Vacancy skills:",
        result["vacancy_skills"]
        if result["vacancy_skills"]
        else "None detected"
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


def main():

    print("=" * 70)
    print("AUTO-HR CV MATCHING SYSTEM")
    print("=" * 70)

    # --------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            'python -m ML1.match_cv '
            '"data/resumes/raw/example.pdf"'
        )

        sys.exit(1)

    cv_path = Path(sys.argv[1])

    if not cv_path.exists():

        print(
            f"\nCV file not found:\n{cv_path}"
        )

        sys.exit(1)

    # --------------------------------------------------
    # Parse CV
    # --------------------------------------------------

    print(f"\nReading CV...")

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

    resume_text = cv["text"]
    resume_skills = cv["skills"]

    if not resume_text.strip():

        print(
            "\nNo text could be extracted "
            "from this CV."
        )

        print(
            "This PDF may require OCR."
        )

        sys.exit(1)

    print("CV successfully parsed.")

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
            print(f"  - {skill}")

    else:
        print("  None")

    # --------------------------------------------------
    # Load matcher
    # --------------------------------------------------

    print("\nLoading job matcher...")

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
        resume_text,
        top_k=10
    )

    # --------------------------------------------------
    # Results
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


if __name__ == "__main__":
    main()
