import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from NLP.config import RESUMES_FILE
from NLP.resume_loader import load_resumes


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = RESUMES_FILE

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "possible_duplicates.csv"
)

NEAR_DUPLICATE_THRESHOLD = 0.85


# ============================================================
# Helpers
# ============================================================

def normalize_for_duplicate_check(text):
    """
    Normalize CV text for duplicate detection.

    This intentionally ignores differences in:
    - capitalization
    - whitespace
    - punctuation
    - anonymization placeholders
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove anonymization placeholders.
    text = re.sub(
        r"\[(email|phone|url|contact)\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Keep letters and numbers.
    text = re.sub(
        r"[^a-zа-яёәіңғүұқөһ0-9]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def create_hash(text):
    """
    SHA-256 fingerprint for exact duplicate detection.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def text_similarity(text_a, text_b):
    """
    Character/sequence similarity.

    Unlike semantic embeddings, this measures
    how similar the actual CV text is.
    """

    return SequenceMatcher(
        None,
        text_a,
        text_b,
    ).ratio()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("AUTO-HR CV DUPLICATE CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    print(
        f"\nReading resumes from:\n"
        f"{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = load_resumes(INPUT_FILE)

    print(
        f"\nLoaded {len(df)} resumes."
    )

    # --------------------------------------------------------
    # Ignore unreadable CVs
    # --------------------------------------------------------

    df = df[
        df["text_length"] > 0
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        f"Usable resumes: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize texts
    # --------------------------------------------------------

    normalized_texts = [
        normalize_for_duplicate_check(text)
        for text in df["text"].fillna("")
    ]

    hashes = [
        create_hash(text)
        for text in normalized_texts
    ]

    duplicates = []

    # --------------------------------------------------------
    # Compare every pair
    # --------------------------------------------------------

    for i in range(
        len(df)
    ):

        for j in range(
            i + 1,
            len(df),
        ):

            # Exact duplicate.
            if hashes[i] == hashes[j]:

                duplicates.append(
                    {
                        "resume_id_1":
                            df.iloc[i]["resume_id"],

                        "filename_1":
                            df.iloc[i]["filename"],

                        "resume_id_2":
                            df.iloc[j]["resume_id"],

                        "filename_2":
                            df.iloc[j]["filename"],

                        "type":
                            "EXACT",

                        "similarity":
                            1.0,
                    }
                )

                continue

            similarity = (
                text_similarity(
                    normalized_texts[i],
                    normalized_texts[j],
                )
            )

            if (
                similarity
                >= NEAR_DUPLICATE_THRESHOLD
            ):

                duplicates.append(
                    {
                        "resume_id_1":
                            df.iloc[i]["resume_id"],

                        "filename_1":
                            df.iloc[i]["filename"],

                        "resume_id_2":
                            df.iloc[j]["resume_id"],

                        "filename_2":
                            df.iloc[j]["filename"],

                        "type":
                            "NEAR",

                        "similarity":
                            similarity,
                    }
                )

    duplicates.sort(
        key=lambda item:
            item["similarity"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("POSSIBLE DUPLICATES")
    print("=" * 70)

    if not duplicates:

        print(
            "\nNo exact or near "
            "duplicates found."
        )

    else:

        for number, pair in enumerate(
            duplicates,
            start=1,
        ):

            print(
                f"\n#{number}"
            )

            print(
                f"Type: "
                f"{pair['type']}"
            )

            print(
                f"Text similarity: "
                f"{pair['similarity']:.2%}"
            )

            print(
                f"CV 1: "
                f"{pair['filename_1']}"
            )

            print(
                f"CV 2: "
                f"{pair['filename_2']}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "resume_id_1",
        "filename_1",
        "resume_id_2",
        "filename_2",
        "type",
        "similarity",
    ]

    report_df = pd.DataFrame(
        duplicates,
        columns=output_columns,
    )

    report_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)

    print(
        f"Duplicate report saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main() 