import argparse

from ML1.cli import add_matching_arguments, matching_config, match_processed_resume


def print_result(
    position,
    result
):

    print("\n" + "=" * 70)

    print(
        f"#{position} "
        f"{result['title']}"
    )

    print("=" * 70)

    print(
        f"City: "
        f"{result['city']}"
    )

    print(
        f"Category: "
        f"{result['job_category']}"
    )

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

    matched = result[
        "matched_skills"
    ]

    missing = result[
        "missing_skills"
    ]

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


def make_parser():
    parser = argparse.ArgumentParser(description="Match pasted resume text against vacancies.")
    add_matching_arguments(parser)
    return parser


def main(argv=None):
    parser = make_parser()
    config = matching_config(parser, parser.parse_args(argv))

    print("=" * 70)
    print("AUTO-HR JOB MATCHING PROTOTYPE")
    print("=" * 70)

    print(
        "\nPaste/type resume text."
    )

    print(
        "When finished, enter an empty line.\n"
    )

    lines = []

    while True:

        line = input()

        if not line.strip():
            break

        lines.append(line)

    resume_text = "\n".join(
        lines
    )

    if not resume_text.strip():

        print(
            "Resume text is empty."
        )

        parser.exit(1)

    print("\nSearching for jobs...")

    try:
        results = match_processed_resume(resume_text, config=config)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Matching failed: {exc}\n")

    print(f"\n\nTOP {len(results)} MATCHING VACANCIES")

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
