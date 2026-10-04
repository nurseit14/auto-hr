import pandas as pd
from pathlib import Path

from NLP.skill_extractor import extract_skills


# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT_DIR / "data" / "processed" / "vacancies_clean.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "vacancies_with_skills.csv"


def main():

    print("Loading vacancies...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} vacancies.")

    # Make sure we have text to process
    if "text" not in df.columns:
        raise ValueError(
            "Column 'text' was not found in vacancies_clean.csv"
        )

    print("Extracting skills...")

    df["skills"] = df["text"].fillna("").apply(extract_skills)

    # Number of extracted skills
    df["skill_count"] = df["skills"].apply(len)

    # Store list nicely in CSV
    df["skills"] = df["skills"].apply(
        lambda skills: ", ".join(skills)
    )

    # Save result
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Finished!")
    print(f"Saved to: {OUTPUT_FILE}")
    print()

    # Statistics
    vacancies_with_skills = (df["skill_count"] > 0).sum()
    vacancies_without_skills = (df["skill_count"] == 0).sum()

    print("Statistics:")
    print(f"Total vacancies: {len(df)}")
    print(f"With detected skills: {vacancies_with_skills}")
    print(f"Without detected skills: {vacancies_without_skills}")

    print()
    print("Examples:")
    print()

    columns = [
        "title",
        "skills",
        "skill_count"
    ]

    print(
        df[columns]
        .sort_values("skill_count", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
