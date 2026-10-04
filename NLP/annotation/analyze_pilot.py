import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "pilot_annotations.json"
)


VALID_LABELS = {
    "HARD_SKILL",
    "SOFT_SKILL",
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
}


def get_annotation_results(task):
    """
    Extract Label Studio annotation results from one task.
    """

    annotations = task.get("annotations", [])

    if not annotations:
        return []

    # For the pilot we expect one completed annotation
    # from the current annotator.
    annotation = annotations[-1]

    return annotation.get("result", [])


def main():

    print("=" * 70)
    print("AUTO-HR PILOT ANNOTATION ANALYSIS")
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        tasks = json.load(file)

    print(f"\nTasks in export: {len(tasks)}")

    annotated_documents = 0
    total_entities = 0

    label_counts = Counter()

    examples = defaultdict(list)

    document_statistics = []

    invalid_labels = Counter()

    # --------------------------------------------------
    # Process tasks
    # --------------------------------------------------

    for task in tasks:

        results = get_annotation_results(task)

        entities = []

        for result in results:

            value = result.get("value", {})

            labels = value.get("labels", [])

            if not labels:
                continue

            label = labels[0]

            text = value.get("text", "")

            start = value.get("start")
            end = value.get("end")

            if label not in VALID_LABELS:
                invalid_labels[label] += 1
                continue

            entities.append(
                {
                    "label": label,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

            label_counts[label] += 1
            total_entities += 1

            if (
                text
                and text not in examples[label]
                and len(examples[label]) < 15
            ):
                examples[label].append(text)

        if entities:

            annotated_documents += 1

            per_document = Counter(
                entity["label"]
                for entity in entities
            )

            resume_id = (
                task.get("data", {})
                .get("resume_id", "UNKNOWN")
            )

            document_statistics.append(
                {
                    "resume_id": resume_id,
                    "total": len(entities),
                    **{
                        label: per_document.get(label, 0)
                        for label in VALID_LABELS
                    }
                }
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nAnnotated documents: "
        f"{annotated_documents}"
    )

    print(
        f"Total entities: "
        f"{total_entities}"
    )

    if annotated_documents:

        average = (
            total_entities
            / annotated_documents
        )

        print(
            f"Average entities/document: "
            f"{average:.2f}"
        )

    # --------------------------------------------------
    # Label distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ENTITY DISTRIBUTION")
    print("=" * 70)

    for label in [
        "HARD_SKILL",
        "SOFT_SKILL",
        "EXPERIENCE",
        "EDUCATION",
        "LANG",
    ]:

        count = label_counts[label]

        percentage = (
            count / total_entities * 100
            if total_entities
            else 0
        )

        print(
            f"{label:<15} "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------
    # Examples
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ENTITY EXAMPLES")
    print("=" * 70)

    for label in [
        "HARD_SKILL",
        "SOFT_SKILL",
        "EXPERIENCE",
        "EDUCATION",
        "LANG",
    ]:

        print(f"\n{label}:")

        if not examples[label]:

            print("  No examples")

            continue

        for text in examples[label]:

            print(f"  - {text}")

    # --------------------------------------------------
    # Per-document statistics
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DOCUMENT STATISTICS")
    print("=" * 70)

    for item in document_statistics:

        print(
            f"\n{item['resume_id']}"
        )

        print(
            f"  Total:      {item['total']}"
        )

        print(
            f"  Hard skill: {item['HARD_SKILL']}"
        )

        print(
            f"  Soft skill: {item['SOFT_SKILL']}"
        )

        print(
            f"  Experience: {item['EXPERIENCE']}"
        )

        print(
            f"  Education:  {item['EDUCATION']}"
        )

        print(
            f"  Language:   {item['LANG']}"
        )

    # --------------------------------------------------
    # Invalid labels
    # --------------------------------------------------

    if invalid_labels:

        print("\n" + "=" * 70)
        print("WARNING: UNKNOWN LABELS")
        print("=" * 70)

        for label, count in invalid_labels.items():

            print(
                f"{label}: {count}"
            )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
