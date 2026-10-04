from pathlib import Path

import numpy as np
import pandas as pd

from ML1.embeddings import encode_texts


ROOT_DIR = Path(__file__).resolve().parent.parent

VACANCIES_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "vacancies_with_skills.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "vacancy_embeddings.npy"
)


def main():

    print("=" * 60)
    print("BUILDING VACANCY EMBEDDINGS")
    print("=" * 60)

    print("\nLoading vacancies...")

    df = pd.read_csv(VACANCIES_FILE)

    print(f"Loaded {len(df)} vacancies.")

    if "text" not in df.columns:
        raise ValueError(
            "Column 'text' not found in vacancy dataset."
        )

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nGenerating semantic embeddings...")

    embeddings = encode_texts(
        texts,
        batch_size=32
    )

    print("\nEmbeddings generated.")
    print(f"Shape: {embeddings.shape}")

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    print(f"\nSaved embeddings to:")
    print(OUTPUT_FILE)

    print("\nChecking saved file...")

    loaded_embeddings = np.load(OUTPUT_FILE)

    print(
        f"Loaded embedding shape: "
        f"{loaded_embeddings.shape}"
    )

    if len(loaded_embeddings) != len(df):
        raise ValueError(
            "Number of embeddings does not match "
            "number of vacancies."
        )

    print("\nEverything looks correct.")


if __name__ == "__main__":
    main()
