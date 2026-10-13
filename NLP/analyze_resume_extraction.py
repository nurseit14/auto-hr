import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "resumes.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "resume_extraction_report.csv"
)


def spaced_letter_score(text):
    """
    Count suspicious spaced-letter sequences.

    Examples:
        P y t h o n
        S K I L L S
        Б І Л І М І
    """

    if not isinstance(text, str):
        return 0

    pattern = (
        r"\b(?:[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]\s+){3,}"
        r"[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]\b"
    )

    matches = re.findall(
        pattern,
        text
    )

    return len(matches)


def find_spaced_letter_sequences(text):
    """
    Return the actual suspicious spaced-letter fragments.
    """

    if not isinstance(text, str):
        return []

    pattern = (
        r"\b(?:[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]\s+){3,}"
        r"[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]\b"
    )

    return re.findall(
        pattern,
        text
    )


def duplicate_line_score(text):
    """
    Count duplicated non-empty lines inside one CV.
    """

    if not isinstance(text, str):
        return 0

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0

    duplicate_count = (
        len(lines) - len(set(lines))
    )

    return duplicate_count


def duplicate_ratio(text):
    """
    Calculate duplicated-line ratio.
    """

    if not isinstance(text, str):
        return 0.0

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    duplicate_count = (
        len(lines) - len(set(lines))
    )

    return (
        duplicate_count
        / len(lines)
    )


def main():

    print("=" * 70)
    print("AUTO-HR RESUME EXTRACTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # Check input file
    # --------------------------------------------------

    if not INPUT_FILE.exists():

        print(
            f"\nInput file not found:\n"
            f"{INPUT_FILE}"
        )

        return

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nLoaded {len(df)} resumes."
    )

    required_columns = [
        "resume_id",
        "filename",
        "raw_text",
        "clean_text",
        "text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------
    # Text lengths
    # --------------------------------------------------

    df["raw_length"] = (
        df["raw_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["clean_length"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["final_length"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    # --------------------------------------------------
    # Cleaning difference
    # --------------------------------------------------

    df["cleaning_difference"] = (
        df["raw_length"]
        - df["clean_length"]
    )

    # --------------------------------------------------
    # Spaced-letter analysis
    #
    # IMPORTANT:
    # Check CLEAN TEXT, not RAW TEXT.
    # --------------------------------------------------

    df["spaced_letter_sequences"] = (
        df["clean_text"]
        .fillna("")
        .apply(spaced_letter_score)
    )

    # --------------------------------------------------
    # Duplicate analysis
    #
    # Also check CLEAN TEXT.
    # --------------------------------------------------

    df["duplicate_lines"] = (
        df["clean_text"]
        .fillna("")
        .apply(duplicate_line_score)
    )

    df["duplicate_ratio"] = (
        df["clean_text"]
        .fillna("")
        .apply(duplicate_ratio)
    )

    # --------------------------------------------------
    # Flags
    # --------------------------------------------------

    df["flag_empty_text"] = (
        df["raw_length"] == 0
    )

    df["flag_short_text"] = (
        df["clean_length"] < 300
    )

    df["flag_spaced_letters"] = (
        df["spaced_letter_sequences"] > 0
    )

    df["flag_duplicates"] = (
        df["duplicate_ratio"] > 0.30
    )

    df["flag_large_cleaning_change"] = (
        (
            df["raw_length"] > 0
        )
        &
        (
            (
                df["cleaning_difference"].abs()
                / df["raw_length"]
            )
            > 0.40
        )
    )

    # --------------------------------------------------
    # Total problem count
    # --------------------------------------------------

    df["problem_count"] = (
        df[
            [
                "flag_empty_text",
                "flag_short_text",
                "flag_spaced_letters",
                "flag_duplicates",
                "flag_large_cleaning_change",
            ]
        ]
        .sum(axis=1)
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n--- SUMMARY ---")

    print(
        f"Empty text: "
        f"{int(df['flag_empty_text'].sum())}"
    )

    print(
        f"Short text: "
        f"{int(df['flag_short_text'].sum())}"
    )

    print(
        f"Spaced-letter problems: "
        f"{int(df['flag_spaced_letters'].sum())}"
    )

    print(
        f"Possible duplicated text: "
        f"{int(df['flag_duplicates'].sum())}"
    )

    print(
        f"Large cleaning changes: "
        f"{int(df['flag_large_cleaning_change'].sum())}"
    )

    # --------------------------------------------------
    # Suspicious resumes
    # --------------------------------------------------

    suspicious = df[
        df["problem_count"] > 0
    ].copy()

    suspicious = suspicious.sort_values(
        by=[
            "problem_count",
            "duplicate_ratio",
            "spaced_letter_sequences",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print("\n" + "=" * 70)
    print("SUSPICIOUS RESUMES")
    print("=" * 70)

    if suspicious.empty:

        print(
            "\nNo suspicious resumes detected."
        )

    else:

        for _, row in suspicious.iterrows():

            print("\n" + "-" * 70)

            print(
                f"Resume ID: "
                f"{row['resume_id']}"
            )

            print(
                f"File: "
                f"{row['filename']}"
            )

            print(
                f"Raw length: "
                f"{row['raw_length']}"
            )

            print(
                f"Clean length: "
                f"{row['clean_length']}"
            )

            print(
                f"Final length: "
                f"{row['final_length']}"
            )

            print(
                f"Cleaning difference: "
                f"{row['cleaning_difference']}"
            )

            print(
                f"Spaced-letter sequences "
                f"after cleaning: "
                f"{row['spaced_letter_sequences']}"
            )

            if row["spaced_letter_sequences"] > 0:

                sequences = find_spaced_letter_sequences(
                    str(row["clean_text"])
                )

                print(
                    "Remaining spaced-letter fragments:"
                )

                for sequence in sequences:

                    print(
                        f"  -> {sequence!r}"
                    )

            print(
                f"Duplicate lines "
                f"after cleaning: "
                f"{row['duplicate_lines']}"
            )

            print(
                f"Duplicate ratio: "
                f"{row['duplicate_ratio']:.2f}"
            )

            print(
                f"Large cleaning change: "
                f"{row['flag_large_cleaning_change']}"
            )

            print(
                f"Problem count: "
                f"{row['problem_count']}"
            )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS FINISHED")
    print("=" * 70)

    print(
        f"\nReport saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()