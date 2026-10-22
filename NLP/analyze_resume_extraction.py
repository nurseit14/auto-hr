import re
from pathlib import Path

import pandas as pd

from NLP.config import RESUMES_FILE
from NLP.resume_loader import load_resumes


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = RESUMES_FILE

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "resume_extraction_report.csv"
)


# ============================================================
# Helpers
# ============================================================

def safe_text(value):
    """
    Convert a dataframe value to safe string.
    """

    if pd.isna(value):
        return ""

    return str(value)


def non_whitespace_length(text):
    """
    Count content characters while ignoring whitespace.

    This is more useful than ordinary string length when
    evaluating cleaning quality, because PDF extraction often
    contains large amounts of layout whitespace.
    """

    text = safe_text(text)

    return len(
        re.sub(
            r"\s+",
            "",
            text,
        )
    )


# ============================================================
# Spaced-letter analysis
# ============================================================

SPACED_LETTER_PATTERN = re.compile(
    r"\b"
    r"(?:"
    r"[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]"
    r"[ \t]+"
    r"){3,}"
    r"[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]"
    r"\b"
)


def spaced_letter_score(text):
    """
    Count suspicious letter-spaced sequences.

    Important:
    this should be applied to FINAL text.

    Example:

        M U R A T ...
        -> if anonymizer already replaced it with [NAME],
           it is no longer a final dataset problem.
    """

    text = safe_text(text)

    return len(
        SPACED_LETTER_PATTERN.findall(
            text
        )
    )


def find_spaced_letter_sequences(text):
    """
    Return suspicious spaced-letter fragments
    from final text.
    """

    text = safe_text(text)

    return (
        SPACED_LETTER_PATTERN.findall(
            text
        )
    )


# ============================================================
# Duplicate analysis
# ============================================================

def normalized_lines(text):
    """
    Return non-empty stripped lines.
    """

    text = safe_text(text)

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def duplicate_line_score(text):
    """
    Count repeated non-empty lines.
    """

    lines = normalized_lines(
        text
    )

    if not lines:
        return 0

    return (
        len(lines)
        - len(set(lines))
    )


def duplicate_ratio(text):
    """
    Calculate duplicated-line ratio.
    """

    lines = normalized_lines(
        text
    )

    if not lines:
        return 0.0

    duplicate_count = (
        len(lines)
        - len(set(lines))
    )

    return (
        duplicate_count
        / len(lines)
    )


# ============================================================
# Content-loss calculation
# ============================================================

def calculate_content_loss(
    raw_text,
    clean_text,
):
    """
    Compare RAW and CLEAN text using only
    non-whitespace characters.

    0.00 means no content-character loss.

    Positive values mean characters disappeared.

    Negative values are possible if controlled cleaning
    reconstructs characters/structure. They are not treated
    as content loss.
    """

    raw_length = non_whitespace_length(
        raw_text
    )

    clean_length = non_whitespace_length(
        clean_text
    )

    if raw_length == 0:
        return 0.0

    loss = (
        raw_length
        - clean_length
    ) / raw_length

    return max(
        0.0,
        loss,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("AUTO-HR RESUME EXTRACTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print(
            f"\nInput file not found:\n"
            f"{INPUT_FILE}"
        )

        return

    df = load_resumes(INPUT_FILE)

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
            + ", ".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Safe text columns
    # --------------------------------------------------------

    for column in [
        "raw_text",
        "clean_text",
        "text",
    ]:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    # --------------------------------------------------------
    # Ordinary character lengths
    # --------------------------------------------------------

    df["raw_length"] = (
        df["raw_text"]
        .str.len()
    )

    df["clean_length"] = (
        df["clean_text"]
        .str.len()
    )

    df["final_length"] = (
        df["text"]
        .str.len()
    )

    df["cleaning_difference"] = (
        df["raw_length"]
        - df["clean_length"]
    )

    # --------------------------------------------------------
    # Non-whitespace lengths
    # --------------------------------------------------------

    df["raw_non_ws_length"] = (
        df["raw_text"]
        .apply(
            non_whitespace_length
        )
    )

    df["clean_non_ws_length"] = (
        df["clean_text"]
        .apply(
            non_whitespace_length
        )
    )

    # --------------------------------------------------------
    # Real content loss
    # --------------------------------------------------------

    df["content_loss_ratio"] = df.apply(
        lambda row:
            calculate_content_loss(
                row["raw_text"],
                row["clean_text"],
            ),
        axis=1,
    )

    # --------------------------------------------------------
    # Spaced-letter analysis
    #
    # Check FINAL text.
    #
    # If a spaced personal name has already been safely
    # anonymized to [NAME], it is not a remaining dataset
    # quality problem.
    # --------------------------------------------------------

    df["spaced_letter_sequences"] = (
        df["text"]
        .apply(
            spaced_letter_score
        )
    )

    # --------------------------------------------------------
    # Duplicate analysis
    #
    # Check CLEAN text because anonymization can legitimately
    # create repeated placeholders like [EMAIL] or [PHONE].
    # --------------------------------------------------------

    df["duplicate_lines"] = (
        df["clean_text"]
        .apply(
            duplicate_line_score
        )
    )

    df["duplicate_ratio"] = (
        df["clean_text"]
        .apply(
            duplicate_ratio
        )
    )

    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    df["flag_empty_text"] = (
        df["final_length"] == 0
    )

    df["flag_short_text"] = (
        df["final_length"] < 300
    )

    df["flag_spaced_letters"] = (
        df["spaced_letter_sequences"] > 0
    )

    df["flag_duplicates"] = (
        df["duplicate_ratio"] > 0.30
    )

    # --------------------------------------------------------
    # Important:
    #
    # Old version used ordinary raw/clean character lengths.
    # That incorrectly treated removal of PDF layout spaces
    # as content loss.
    #
    # Now flag only when >10% of NON-WHITESPACE characters
    # disappear.
    # --------------------------------------------------------

    df["flag_content_loss"] = (
        df["content_loss_ratio"] > 0.10
    )

    # Keep old column name for compatibility with previous
    # reports/scripts, but give it the corrected meaning.
    df["flag_large_cleaning_change"] = (
        df["flag_content_loss"]
    )

    # --------------------------------------------------------
    # Total problem count
    # --------------------------------------------------------

    problem_columns = [
        "flag_empty_text",
        "flag_short_text",
        "flag_spaced_letters",
        "flag_duplicates",
        "flag_content_loss",
    ]

    df["problem_count"] = (
        df[
            problem_columns
        ]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n--- SUMMARY ---"
    )

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
        f"Real content-loss problems: "
        f"{int(df['flag_content_loss'].sum())}"
    )

    # --------------------------------------------------------
    # Suspicious resumes
    # --------------------------------------------------------

    suspicious = df[
        df["problem_count"] > 0
    ].copy()

    suspicious = (
        suspicious
        .sort_values(
            by=[
                "problem_count",
                "content_loss_ratio",
                "duplicate_ratio",
                "spaced_letter_sequences",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUSPICIOUS RESUMES"
    )

    print(
        "=" * 70
    )

    if suspicious.empty:

        print(
            "\nNo suspicious resumes detected."
        )

    else:

        for _, row in suspicious.iterrows():

            print(
                "\n"
                + "-" * 70
            )

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
                f"Raw non-whitespace length: "
                f"{row['raw_non_ws_length']}"
            )

            print(
                f"Clean non-whitespace length: "
                f"{row['clean_non_ws_length']}"
            )

            print(
                f"Content loss: "
                f"{row['content_loss_ratio']:.2%}"
            )

            print(
                f"Spaced-letter sequences "
                f"in final text: "
                f"{row['spaced_letter_sequences']}"
            )

            if (
                row[
                    "spaced_letter_sequences"
                ]
                > 0
            ):

                sequences = (
                    find_spaced_letter_sequences(
                        row["text"]
                    )
                )

                print(
                    "Remaining spaced-letter "
                    "fragments:"
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
                f"Real content-loss flag: "
                f"{row['flag_content_loss']}"
            )

            print(
                f"Problem count: "
                f"{row['problem_count']}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANALYSIS FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReport saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()