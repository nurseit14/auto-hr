import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"

HUMAN_FILE = (
    ANNOTATIONS_DIR
    / "human_reviewed"
    / "unseen_0006_0010.json"
)

PREDICTION_FILES = [
    ANNOTATIONS_DIR / "resume_hybrid_preannotated_tasks.json",
    ANNOTATIONS_DIR / "hybrid_0007_0010.json",
]

LABELS = [
    "HARD_SKILL",
    "SOFT_SKILL",
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
]


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def extract_entities(results):
    """
    Convert Label Studio regions into:
        (start, end, label)

    Evaluation is strict exact entity matching.
    """

    entities = set()

    for region in results:
        if not isinstance(region, dict):
            continue

        value = region.get("value", {})

        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])

        if start is None or end is None:
            continue

        if not labels:
            continue

        for label in labels:
            if label not in LABELS:
                continue

            entities.add(
                (
                    int(start),
                    int(end),
                    label,
                )
            )

    return entities


def get_resume_id(task):
    return str(
        task.get("data", {}).get(
            "resume_id",
            f"task_{task.get('id', 'unknown')}",
        )
    )


def load_original_predictions():
    """
    Load the original Hybrid V1 predictions generated
    BEFORE human review.

    Returns:
        {
            "resume_0006": set(...),
            ...
        }
    """

    predictions_by_resume = {}

    for path in PREDICTION_FILES:
        if not path.exists():
            print(f"WARNING: prediction file missing: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        print(f"Loading predictions: {path.name}")
        print(f"  Tasks found: {len(tasks)}")

        for task in tasks:
            resume_id = get_resume_id(task)

            predictions = task.get("predictions", [])

            if not predictions:
                continue

            valid_predictions = [
                p
                for p in predictions
                if isinstance(p, dict)
            ]

            if not valid_predictions:
                continue

            selected = None

            for prediction in valid_predictions:
                if (
                    prediction.get("model_version")
                    == "autohr-hybrid-v1"
                ):
                    selected = prediction
                    break

            if selected is None:
                selected = valid_predictions[0]

            entities = extract_entities(
                selected.get("result", [])
            )

            predictions_by_resume[resume_id] = entities

    return predictions_by_resume


def load_human_annotations():
    """
    Load submitted human-reviewed Label Studio annotations.
    """

    if not HUMAN_FILE.exists():
        raise FileNotFoundError(
            f"Human-reviewed file not found:\n{HUMAN_FILE}"
        )

    with open(HUMAN_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    annotations_by_resume = {}

    for task in tasks:
        resume_id = get_resume_id(task)

        annotations = task.get("annotations", [])

        valid_annotations = [
            annotation
            for annotation in annotations
            if isinstance(annotation, dict)
            and not annotation.get("was_cancelled", False)
        ]

        if not valid_annotations:
            continue

        # Use latest submitted annotation.
        annotation = valid_annotations[-1]

        entities = extract_entities(
            annotation.get("result", [])
        )

        annotations_by_resume[resume_id] = entities

    return annotations_by_resume


def main():
    print("=" * 72)
    print("AUTO-HR HYBRID V1 — UNSEEN EVALUATION")
    print("=" * 72)

    print("\nLoading ORIGINAL Hybrid V1 predictions...\n")

    predictions = load_original_predictions()

    print("\nLoading HUMAN-REVIEWED annotations...\n")

    humans = load_human_annotations()

    print(f"Original prediction CVs loaded: {len(predictions)}")
    print(f"Human-reviewed CVs loaded:      {len(humans)}")

    common_ids = sorted(
        set(predictions.keys())
        & set(humans.keys())
    )

    print(f"CVs available for evaluation:   {len(common_ids)}")

    print("\nMatched CVs:")

    for resume_id in common_ids:
        print(f"  ✓ {resume_id}")

    missing_predictions = sorted(
        set(humans.keys())
        - set(predictions.keys())
    )

    if missing_predictions:
        print("\nHuman CVs missing original predictions:")

        for resume_id in missing_predictions:
            print(f"  ✗ {resume_id}")

    if not common_ids:
        raise RuntimeError(
            "No matching resume IDs were found between "
            "predictions and human annotations."
        )

    label_stats = defaultdict(
        lambda: Counter(
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
        )
    )

    document_results = []

    for resume_id in common_ids:
        predicted = predictions[resume_id]
        human = humans[resume_id]

        tp_entities = predicted & human
        fp_entities = predicted - human
        fn_entities = human - predicted

        for _, _, label in tp_entities:
            label_stats[label]["tp"] += 1

        for _, _, label in fp_entities:
            label_stats[label]["fp"] += 1

        for _, _, label in fn_entities:
            label_stats[label]["fn"] += 1

        tp = len(tp_entities)
        fp = len(fp_entities)
        fn = len(fn_entities)

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
        )

        document_results.append(
            {
                "resume_id": resume_id,
                "human": len(human),
                "predicted": len(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    print("\n" + "=" * 72)
    print("HYBRID V1 VS HUMAN GROUND TRUTH — UNSEEN SET")
    print("=" * 72)

    print(
        f"\n{'Label':<16}"
        f"{'TP':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 72)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in LABELS:
        tp = label_stats[label]["tp"]
        fp = label_stats[label]["fp"]
        fn = label_stats[label]["fn"]

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"{label:<16}"
            f"{tp:>7}"
            f"{fp:>7}"
            f"{fn:>7}"
            f"{precision:>12.3f}"
            f"{recall:>10.3f}"
            f"{f1:>10.3f}"
        )

    micro_precision, micro_recall, micro_f1 = (
        calculate_metrics(
            total_tp,
            total_fp,
            total_fn,
        )
    )

    print("-" * 72)

    print(
        f"{'OVERALL':<16}"
        f"{total_tp:>7}"
        f"{total_fp:>7}"
        f"{total_fn:>7}"
        f"{micro_precision:>12.3f}"
        f"{micro_recall:>10.3f}"
        f"{micro_f1:>10.3f}"
    )

    print("\n" + "=" * 72)
    print("PER-DOCUMENT RESULTS")
    print("=" * 72)

    for result in document_results:
        print(f"\n{result['resume_id']}")

        print(
            f"  Human entities:  {result['human']}"
        )

        print(
            f"  Hybrid entities: {result['predicted']}"
        )

        print(
            f"  TP: {result['tp']}  "
            f"FP: {result['fp']}  "
            f"FN: {result['fn']}"
        )

        print(
            f"  Precision: {result['precision']:.3f}"
        )

        print(
            f"  Recall:    {result['recall']:.3f}"
        )

        print(
            f"  F1:        {result['f1']:.3f}"
        )

    mean_document_f1 = (
        sum(
            result["f1"]
            for result in document_results
        )
        / len(document_results)
    )

    development_f1 = 0.7288

    difference = (
        micro_f1
        - development_f1
    )

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(f"\nEvaluation CVs:    {len(common_ids)}")
    print("Evaluation set:    UNSEEN #0006–#0010")
    print("Method:            HYBRID V1")

    print(f"\nTrue positives:    {total_tp}")
    print(f"False positives:   {total_fp}")
    print(f"False negatives:   {total_fn}")

    print(
        f"\nMicro Precision:   "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall:      "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1:          "
        f"{micro_f1:.4f}"
    )

    print(
        f"Mean document F1:  "
        f"{mean_document_f1:.4f}"
    )

    print("\nDevelopment-set result:")
    print("  Precision: 0.6692")
    print("  Recall:    0.8000")
    print("  F1:        0.7288")

    print("\nUnseen-set result:")
    print(
        f"  Precision: {micro_precision:.4f}"
    )
    print(
        f"  Recall:    {micro_recall:.4f}"
    )
    print(
        f"  F1:        {micro_f1:.4f}"
    )

    print(
        f"\nF1 difference: "
        f"{difference:+.4f}"
    )

    print(
        "\nEvaluation type: strict entity-level exact match"
    )

    print(
        "Correct = identical start offset + end offset + label."
    )


if __name__ == "__main__":
    main()