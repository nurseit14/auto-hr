import json
from pathlib import Path

import pandas as pd
from NLP.config import RESUMES_FILE
from NLP.resume_loader import load_resumes


ROOT_DIR = Path(__file__).resolve().parent.parent.parent



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

    df = load_resumes(RESUMES_FILE)

    print(
        f"\nLoaded {len(df)} resumes."
    )

    # ----------------------------------------
    # Remove CVs without extractable text
    # ----------------------------------------

    df = df[
        df["text_length"] > 0
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

        # Preserve exact frozen text for NER offsets.
        text = str(text)

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

                "document_id":
                    str(row["document_id"]),

                "text_sha256":
                    str(row["text_sha256"]),

                "text_version":
                    str(row.get("text_version", "v1")),
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