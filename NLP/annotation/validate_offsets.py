import argparse
import json
from pathlib import Path

import pandas as pd

from NLP.config import RESUMES_FILE
from NLP.resume_loader import load_resumes


ROOT_DIR = Path(__file__).resolve().parent.parent.parent




def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate Label Studio span offsets "
            "against the current resume text."
        )
    )

    parser.add_argument(
        "json_file",
        type=str,
        help="Path to Label Studio JSON file"
    )

    return parser.parse_args()


def get_regions(task):
    """
    Collect span regions from predictions
    and human annotations.
    """

    regions = []

    # Machine predictions
    for prediction in task.get("predictions", []):
        for region in prediction.get("result", []):
            regions.append(
                ("prediction", region)
            )

    # Human annotations
    for annotation in task.get("annotations", []):
        for region in annotation.get("result", []):
            regions.append(
                ("annotation", region)
            )

    return regions


def main():

    args = parse_args()

    json_path = Path(
        args.json_file
    )

    print("=" * 70)
    print("AUTO-HR OFFSET VALIDATOR")
    print("=" * 70)

    # --------------------------------------------------
    # Check files
    # --------------------------------------------------

    if not RESUMES_FILE.exists():
        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUMES_FILE}"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Annotation JSON not found:\n"
            f"{json_path}"
        )

    # --------------------------------------------------
    # Load current resume dataset
    # --------------------------------------------------

    df = load_resumes(RESUMES_FILE)

    current_texts = {}

    for _, row in df.iterrows():

        resume_id = str(
            row["resume_id"]
        )

        text = row.get(
            "text",
            ""
        )

        if pd.isna(text):
            text = ""

        current_texts[
            resume_id
        ] = str(text)

    print(
        f"\nCurrent resumes loaded: "
        f"{len(current_texts)}"
    )

    # --------------------------------------------------
    # Load annotation JSON
    # --------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        tasks = json.load(
            file
        )

    print(
        f"Annotation tasks loaded: "
        f"{len(tasks)}"
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    task_text_changed = 0
    missing_resume_ids = 0

    total_regions = 0
    valid_regions = 0
    invalid_regions = 0

    invalid_examples = []

    # --------------------------------------------------
    # Validate every task
    # --------------------------------------------------

    for task in tasks:

        data = task.get(
            "data",
            {}
        )

        resume_id = str(
            data.get(
                "resume_id",
                ""
            )
        )

        task_text = str(
            data.get(
                "text",
                ""
            )
        )

        # --------------------------------------------------
        # Resume ID check
        # --------------------------------------------------

        if resume_id not in current_texts:

            missing_resume_ids += 1

            print(
                f"\nWARNING: "
                f"{resume_id} not found "
                f"in current resumes.csv"
            )

            continue

        current_text = (
            current_texts[
                resume_id
            ]
        )

        # --------------------------------------------------
        # Check whether task text changed
        # --------------------------------------------------

        if task_text != current_text:

            task_text_changed += 1

            print(
                f"\nTEXT CHANGED: "
                f"{resume_id}"
            )

            print(
                f"  JSON text length: "
                f"{len(task_text)}"
            )

            print(
                f"  Current text length: "
                f"{len(current_text)}"
            )

        # --------------------------------------------------
        # Validate each annotation region
        # --------------------------------------------------

        regions = get_regions(
            task
        )

        for source, region in regions:

            value = region.get(
                "value",
                {}
            )

            start = value.get(
                "start"
            )

            end = value.get(
                "end"
            )

            stored_text = str(
                value.get(
                    "text",
                    ""
                )
            )

            labels = value.get(
                "labels",
                []
            )

            if (
                start is None
                or end is None
            ):
                continue

            total_regions += 1

            start = int(start)
            end = int(end)

            # --------------------------------------------------
            # Check numerical range
            # --------------------------------------------------

            if (
                start < 0
                or end > len(current_text)
                or start >= end
            ):

                invalid_regions += 1

                invalid_examples.append(
                    {
                        "resume_id":
                            resume_id,

                        "source":
                            source,

                        "label":
                            labels,

                        "start":
                            start,

                        "end":
                            end,

                        "stored_text":
                            stored_text,

                        "current_slice":
                            "<INVALID RANGE>",
                    }
                )

                continue

            # --------------------------------------------------
            # Compare offsets with current text
            # --------------------------------------------------

            current_slice = (
                current_text[
                    start:end
                ]
            )

            if (
                current_slice
                == stored_text
            ):

                valid_regions += 1

            else:

                invalid_regions += 1

                invalid_examples.append(
                    {
                        "resume_id":
                            resume_id,

                        "source":
                            source,

                        "label":
                            labels,

                        "start":
                            start,

                        "end":
                            end,

                        "stored_text":
                            stored_text,

                        "current_slice":
                            current_slice,
                    }
                )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OFFSET VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Tasks whose text changed: "
        f"{task_text_changed}"
    )

    print(
        f"Missing resume IDs: "
        f"{missing_resume_ids}"
    )

    print(
        f"Total regions checked: "
        f"{total_regions}"
    )

    print(
        f"Valid regions: "
        f"{valid_regions}"
    )

    print(
        f"Invalid regions: "
        f"{invalid_regions}"
    )

    # --------------------------------------------------
    # Show examples
    # --------------------------------------------------

    if invalid_examples:

        print("\n" + "=" * 70)
        print("INVALID OFFSET EXAMPLES")
        print("=" * 70)

        for example in invalid_examples[:20]:

            print("\n" + "-" * 70)

            print(
                f"Resume: "
                f"{example['resume_id']}"
            )

            print(
                f"Source: "
                f"{example['source']}"
            )

            print(
                f"Label: "
                f"{example['label']}"
            )

            print(
                f"Offsets: "
                f"{example['start']}:"
                f"{example['end']}"
            )

            print(
                f"Stored text: "
                f"{example['stored_text']!r}"
            )

            print(
                f"Current slice: "
                f"{example['current_slice']!r}"
            )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 70)

    if (
        task_text_changed == 0
        and invalid_regions == 0
        and missing_resume_ids == 0
    ):

        print(
            "RESULT: OFFSETS ARE CONSISTENT"
        )

    else:

        print(
            "RESULT: OLD OFFSETS SHOULD "
            "BE REGENERATED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()