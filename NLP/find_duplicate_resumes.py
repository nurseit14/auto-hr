from pathlib import Path

import numpy as np
import pandas as pd

from ML1.embeddings import encode_texts


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "resumes.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "possible_duplicates.csv"
)


# High threshold because two CVs from similar
# students may naturally discuss similar technologies.
DUPLICATE_THRESHOLD = 0.92


def main():

    print("=" * 70)
    print("AUTO-HR RESUME DUPLICATE DETECTOR")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(df)} resumes.")

    # Ignore PDFs from which no text could be extracted.
    usable = df[
        df["character_count"] > 0
    ].copy()

    usable.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"Resumes with usable text: {len(usable)}"
    )

    texts = (
        usable["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nGenerating resume embeddings...")

    embeddings = encode_texts(
        texts,
        batch_size=16
    )

    print(
        f"\nEmbedding shape: "
        f"{embeddings.shape}"
    )

    # Embeddings are normalized.
    # Matrix multiplication therefore gives
    # cosine similarity.
    similarity_matrix = (
        embeddings @ embeddings.T
    )

    duplicate_pairs = []

    for i in range(len(usable)):

        for j in range(i + 1, len(usable)):

            similarity = float(
                similarity_matrix[i, j]
            )

            if similarity >= DUPLICATE_THRESHOLD:

                duplicate_pairs.append(
                    {
                        "resume_id_1":
                            usable.iloc[i]["resume_id"],

                        "filename_1":
                            usable.iloc[i]["filename"],

                        "resume_id_2":
                            usable.iloc[j]["resume_id"],

                        "filename_2":
                            usable.iloc[j]["filename"],

                        "similarity":
                            similarity,
                    }
                )

    duplicate_pairs.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("POSSIBLE DUPLICATES")
    print("=" * 70)

    if not duplicate_pairs:

        print(
            "\nNo possible duplicates found "
            f"above {DUPLICATE_THRESHOLD:.0%}."
        )

    else:

        for number, pair in enumerate(
            duplicate_pairs,
            start=1
        ):

            print(
                f"\n#{number}"
            )

            print(
                f"Similarity: "
                f"{pair['similarity']:.2%}"
            )

            print(
                f"CV 1: {pair['filename_1']}"
            )

            print(
                f"CV 2: {pair['filename_2']}"
            )

    # Save results
    duplicate_df = pd.DataFrame(
        duplicate_pairs
    )

    duplicate_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"Possible duplicate report saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
