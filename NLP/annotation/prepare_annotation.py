import json
from pathlib import Path

import pandas as pd


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
    / "resume_annotation_tasks.json"
)


def main():

    print("=" * 70)
    print("AUTO-HR ANNOTATION DATA PREPARATION")
    print("=" * 70)

    print("\nLoading resumes...")

    df = pd.read_csv(
        RESUMES_FILE
    )

    print(
        f"Loaded {len(df)} resumes."
    )

    # ----------------------------------------
    # Remove CVs without extractable text
    # ----------------------------------------

    df = df[
        df["character_count"] > 0
    ].copy()

    print(
        f"Usable resumes: {len(df)}"
    )

    tasks = []

    for _, row in df.iterrows():

        text = str(
            row["text"]
        ).strip()

        if not text:
            continue

        task = {
            "data": {
                "text": text,

                # Anonymous internal ID
                "resume_id":
                    row["resume_id"]
            }
        }

        tasks.append(task)

    # ----------------------------------------
    # Save Label Studio JSON
    # ----------------------------------------

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
    print("ANNOTATION DATA READY")
    print("=" * 70)

    print(
        f"\nTasks created: "
        f"{len(tasks)}"
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
