import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

RESUMES_FILE = (
    ROOT_DIR
    / "data"
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

    print("\nLoading resumes from:")
    print(RESUMES_FILE)

    # ----------------------------------------
    # Check input file
    # ----------------------------------------

    if not RESUMES_FILE.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUMES_FILE}"
        )

    df = pd.read_csv(
        RESUMES_FILE
    )

    print(
        f"\nLoaded {len(df)} resumes."
    )

    # ----------------------------------------
    # Remove CVs without extractable text
    # ----------------------------------------

    if "character_count" in df.columns:

        df = df[
            df["character_count"] > 0
        ].copy()

    print(
        f"Usable resumes: {len(df)}"
    )

    tasks = []

    # ----------------------------------------
    # Create Label Studio tasks
    # ----------------------------------------

    for _, row in df.iterrows():

        text = row.get(
            "text",
            ""
        )

        if pd.isna(text):
            text = ""

        text = str(
            text
        ).strip()

        if not text:
            continue

        resume_id = str(
            row.get(
                "resume_id",
                ""
            )
        )

        filename = str(
            row.get(
                "filename",
                ""
            )
        )

        task = {
            "data": {
                "text": text,

                # Internal document ID
                "resume_id":
                    resume_id,

                # Useful for manual review
                "filename":
                    filename,
            }
        }

        tasks.append(
            task
        )

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