import argparse
import hashlib
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

    current_documents = {}

    for _, row in df.iterrows():
        document_id = str(row["document_id"])

        if not document_id:
            raise ValueError("Empty document_id in resumes.csv")

        if document_id in current_documents:
            raise ValueError(
                f"Duplicate document_id: {document_id}"
            )

        current_documents[document_id] = {
            "resume_id": str(row["resume_id"]),
            "text": str(row["text"]),
            "source_sha256": str(row["source_sha256"]),
            "text_sha256": str(row["text_sha256"]),
            "text_version": str(row["text_version"]),
        }

    print(
        f"\\nCurrent resumes loaded: "
        f"{len(current_documents)}"
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
    identity_errors = []

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
        # Stable document and text-version validation
        # --------------------------------------------------

        document_id = str(data.get("document_id") or "")

        required = (
            "document_id",
            "source_sha256",
            "text_sha256",
            "text_version",
        )

        missing = [
            field for field in required
            if not data.get(field)
        ]

        if missing:
            status = "MISSING_METADATA"
            detail = ", ".join(missing)
            identity_errors.append((status, resume_id, detail))
            print(f"ERROR [{status}] {resume_id}: {detail}")
            continue

        if document_id not in current_documents:
            status = "UNKNOWN_DOCUMENT"
            identity_errors.append(
                (status, resume_id, document_id)
            )
            print(f"ERROR [{status}] {document_id}")
            continue

        current = current_documents[document_id]
        current_text = current["text"]

        mismatches = []

        if str(data["source_sha256"]) != current["source_sha256"]:
            mismatches.append("SOURCE_HASH_MISMATCH")

        if str(data["text_version"]) != current["text_version"]:
            mismatches.append("VERSION_MISMATCH")

        task_hash = hashlib.sha256(
            task_text.encode("utf-8")
        ).hexdigest()

        if (
            str(data["text_sha256"]) != current["text_sha256"]
            or str(data["text_sha256"]) != task_hash
            or task_text != current_text
        ):
            mismatches.append("TEXT_MISMATCH")
            task_text_changed += 1

        if mismatches:
            for status in mismatches:
                identity_errors.append(
                    (status, resume_id, document_id)
                )
                print(
                    f"ERROR [{status}] "
                    f"{document_id} ({resume_id})"
                )
            continue

        # resume_id is informational only.
        # Renumbering must not invalidate the document.

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
        f"Identity/version errors: "
        f"{len(identity_errors)}"
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
        and not identity_errors
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

    if identity_errors or invalid_regions or task_text_changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()