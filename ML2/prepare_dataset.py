"""Validate saved Label Studio annotations and freeze candidate-grouped datasets.

This command is local and does not call a model, train one, or edit Label Studio.
Predictions and data.model_assessment are deliberately never read as targets.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "ml2-reviewed-pairs-v1"
CATEGORIES = {"ROLE_LEVEL", "EXPERIENCE", "HARD_SKILL", "SOFT_SKILL",
              "EDUCATION", "LANGUAGE", "OTHER"}
STATUS = {0: "unknown", 1: "mismatch", 2: "partial", 3: "match"}
REVIEW_CHOICE = "Я проверил требования, баллы и общую оценку"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", text)


def canonical_quote(quote, source):
    """Restore original whitespace without changing any non-whitespace character."""
    if not quote:
        return ""
    if quote in source:
        return quote
    wanted = compact(quote)
    if not wanted:
        raise ValueError("CV quote is whitespace only")
    positions = [n for n, char in enumerate(source) if not char.isspace()]
    start = "".join(source[n] for n in positions).find(wanted)
    if start < 0:
        raise ValueError("CV quote is not a contiguous source substring")
    return source[positions[start]:positions[start + len(wanted) - 1] + 1]


def single(result, name, key, region_id=None, optional=False):
    items = [r for r in result if r.get("from_name") == name
             and (region_id is None or r.get("id") == region_id)]
    if not items and optional:
        return ""
    if len(items) != 1:
        raise ValueError(f"{name}: expected one saved control, got {len(items)}")
    values = items[0].get("value", {}).get(key, [])
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"{name}: expected one string value")
    value = values[0].strip()
    if not optional and not value:
        raise ValueError(f"{name}: empty value")
    return value


def score_value(value):
    if value not in {"0", "1", "2", "3"}:
        raise ValueError("Score must be 0, 1, 2 or 3")
    return int(value)


def parse_task(task):
    data = task["data"]
    for key in ("pair_id", "cv_id", "vacancy_id", "cv_text", "vacancy_display"):
        if key not in data or not isinstance(data[key], str):
            raise ValueError(f"Missing string data.{key}")
    if not all(data[k].strip() for k in ("pair_id", "cv_id", "vacancy_id")):
        raise ValueError("Pair, CV and vacancy IDs must be nonempty")
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if len(annotations) != 1:
        raise ValueError("Expected one saved annotation; predictions/drafts are not accepted")
    annotation = annotations[0]
    result = annotation["result"]
    if single(result, "human_review", "choices") != REVIEW_CHOICE:
        raise ValueError("Saved review checkbox is missing")
    overall = score_value(single(result, "overall_score", "choices"))
    explanation = single(result, "overall_reason", "text")
    cv, vacancy = data["cv_text"], data["vacancy_display"]
    for field, source in (("cv_text_sha256", cv), ("vacancy_text_sha256", vacancy)):
        if data.get(field) and data[field] != sha256(source):
            raise ValueError(f"{field}: source hash mismatch")
    regions = [r for r in result if r.get("from_name") == "requirement_type"
               and r.get("type") == "labels"]
    if not regions:
        raise ValueError("No saved requirement regions")
    requirements, ids, spans = [], set(), set()
    for region in regions:
        rid, value = region["id"], region["value"]
        start, end = value["start"], value["end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(vacancy)):
            raise ValueError("Invalid vacancy span offsets")
        text = vacancy[start:end]
        if value.get("text") != text:
            raise ValueError("Vacancy span text does not match its offsets")
        if ((start and vacancy[start - 1].isalnum() and vacancy[start].isalnum())
                or (end < len(vacancy) and vacancy[end - 1].isalnum() and vacancy[end].isalnum())):
            raise ValueError("Vacancy span cuts through a word")
        labels = value.get("labels", [])
        if len(labels) != 1 or labels[0] not in CATEGORIES:
            raise ValueError("Invalid requirement category")
        category = labels[0]
        if rid in ids or (start, end, category) in spans:
            raise ValueError("Duplicate requirement ID or span")
        ids.add(rid)
        spans.add((start, end, category))
        title = single(result, "requirement_title", "text", rid)
        score = score_value(single(result, "match_score", "choices", rid))
        reason = single(result, "reason", "text", rid)
        supplied_quote = single(result, "cv_evidence", "text", rid, optional=True)
        if (score == 0 and supplied_quote) or (score > 0 and not supplied_quote):
            raise ValueError("Requirement score and CV evidence disagree")
        quote = canonical_quote(supplied_quote, cv)
        requirements.append({"requirement_id": rid, "category": category,
                             "requirement": title, "vacancy_quote": text,
                             "vacancy_start": start, "vacancy_end": end,
                             "score": score, "status": STATUS[score],
                             "judged": score > 0, "cv_quote": quote,
                             "explanation": reason})
    per_region_names = {"requirement_title", "match_score", "reason", "cv_evidence"}
    if any(r.get("from_name") in per_region_names and r.get("id") not in ids for r in result):
        raise ValueError("Orphan saved requirement control")
    if not cv.strip() and (overall != 0 or any(r["score"] for r in requirements)):
        raise ValueError("An empty CV cannot have judged scores")
    if all(r["score"] == 0 for r in requirements) and overall != 0:
        raise ValueError("All-unknown requirements require unknown overall score")
    if overall == 3 and any(r["category"] == "ROLE_LEVEL" and r["score"] == 1 for r in requirements):
        raise ValueError("Overall good fit contradicts confirmed role mismatch")
    fields = data.get("vacancy_fields", {})
    if isinstance(fields, str):
        fields = json.loads(fields)
    # This is the unmodified source, not annotation-derived requirements or predictions.
    return {"task_id": task["id"], "pair_id": data["pair_id"], "cv_id": data["cv_id"],
            "vacancy_id": data["vacancy_id"], "cv_text": cv, "vacancy_display": vacancy,
            "vacancy_title": data.get("vacancy_title", ""),
            "vacancy_fields": {k: fields.get(k, "") for k in
                               ("title", "experience", "requirements", "responsibilities", "text")},
            "cv_text_sha256": sha256(cv), "vacancy_text_sha256": sha256(vacancy),
            "overall_score": overall, "judged": overall > 0,
            "relevance": overall - 1 if overall else None,
            "overall_explanation": explanation, "requirements": requirements,
            "annotation_source": {"kind": "label_studio_saved_annotation",
                                  "annotation_id": annotation["id"],
                                  "updated_at": annotation.get("updated_at"),
                                  "review_checkbox_present": True},
            "selection": {k: data.get(k) for k in
                          ("selection_method", "selection_rank", "selection_score")}}


def candidate_groups(rows, same_candidate=()):
    texts, vacancies, pairs = {}, {}, set()
    by_cv = defaultdict(list)
    for row in rows:
        cid, vid = row["cv_id"], row["vacancy_id"]
        if row["pair_id"] in pairs:
            raise ValueError("Duplicate pair ID")
        pairs.add(row["pair_id"])
        if cid in texts and texts[cid] != row["cv_text"]:
            raise ValueError(f"Conflicting source texts for CV {cid}")
        if vid in vacancies and vacancies[vid] != row["vacancy_display"]:
            raise ValueError(f"Conflicting source texts for vacancy {vid}")
        texts[cid], vacancies[vid] = row["cv_text"], row["vacancy_display"]
        by_cv[cid].append(row)
    for cid, group in by_cv.items():
        if len(group) != 5 or len({r["vacancy_id"] for r in group}) != 5:
            raise ValueError(f"CV {cid}: expected five distinct vacancy pairs")
    parent = {cid: cid for cid in texts}

    def find(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def union(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a

    exact = {}
    for cid, text in sorted(texts.items()):
        normalized = compact(text).casefold()
        if normalized and normalized in exact:
            union(cid, exact[normalized])
        elif normalized:
            exact[normalized] = cid
    for group in same_candidate:
        if len(group) < 2 or len(group) != len(set(group)) or any(cid not in texts for cid in group):
            raise ValueError("same-candidate needs at least two distinct existing CV IDs")
        for cid in group[1:]:
            union(group[0], cid)
    grouped = defaultdict(list)
    for cid in sorted(texts):
        grouped[find(cid)].append(cid)
    groups = {"candidate_" + sha256("|".join(ids))[:12]: ids for ids in grouped.values()}
    lookup = {cid: gid for gid, ids in groups.items() for cid in ids}
    for row in rows:
        row["candidate_group_id"] = lookup[row["cv_id"]]
    return groups


def split_groups(rows, seed=42, validation_fraction=.15, test_fraction=.15):
    """Seeded group stratification; never optimizes a model or its test results."""
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    eligible = [r for r in rows if r["judged"] and r["cv_text"].strip()]
    by_group = defaultdict(list)
    for row in eligible:
        by_group[row["candidate_group_id"]].append(row)
    group_ids = sorted(by_group)
    if len(group_ids) < 3:
        raise ValueError("Need at least three candidate groups for train/validation/test")
    nv = max(1, round(len(group_ids) * validation_fraction))
    nt = max(1, round(len(group_ids) * test_fraction))
    if nv + nt >= len(group_ids):
        raise ValueError("No candidate group left for training")
    sizes = (len(group_ids) - nv - nt, nv, nt)
    total = Counter(r["overall_score"] for r in eligible)
    groups_per_label = Counter(label for rs in by_group.values()
                               for label in {r["overall_score"] for r in rs})
    rng, best = random.Random(seed), None
    # Different candidates can have different numbers of CV versions. Count both
    # candidate groups and label proportions rather than splitting individual rows.
    for _ in range(1024):
        order = list(group_ids)
        rng.shuffle(order)
        parts, offset, cost = [], 0, 0.
        for size in sizes:
            part = order[offset:offset + size]
            offset += size
            counts = Counter(r["overall_score"] for gid in part for r in by_group[gid])
            fraction = size / len(group_ids)
            cost += sum((counts[label] - count * fraction) ** 2 / (count * fraction + 1)
                        for label, count in total.items())
            cost += sum(10000 for label in total
                        if groups_per_label[label] >= 3 and not counts[label])
            parts.append(part)
        if best is None or cost < best[0]:
            best = cost, parts
    assignment = {gid: split for split, part in zip(("train", "validation", "test"), best[1])
                  for gid in part}
    output = {split: sorted((r for r in eligible if assignment[r["candidate_group_id"]] == split),
                            key=lambda r: r["pair_id"])
              for split in ("train", "validation", "test")}
    return output, assignment


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(export_path, output, same_candidate=(), seed=42,
                  validation_fraction=.15, test_fraction=.15):
    export_path, output = Path(export_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new dataset version")
    source_bytes = export_path.read_bytes()
    tasks = json.loads(source_bytes)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Expected a nonempty full Label Studio JSON export")
    rows = []
    for task in sorted(tasks, key=lambda t: t["id"]):
        try:
            rows.append(parse_task(task))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Task {task.get('id', '?')}: {error}") from error
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    groups = candidate_groups(rows, same_candidate)
    splits, assignment = split_groups(rows, seed, validation_fraction, test_fraction)
    quarantined = [{"pair_id": r["pair_id"], "task_id": r["task_id"], "cv_id": r["cv_id"],
                    "reason": "empty_cv" if not r["cv_text"].strip() else "overall_unknown"}
                   for r in rows if not r["judged"] or not r["cv_text"].strip()]
    requirements = [{"pair_id": r["pair_id"], "task_id": r["task_id"],
                     "cv_id": r["cv_id"], "candidate_group_id": r["candidate_group_id"],
                     "vacancy_id": r["vacancy_id"], **req}
                    for r in rows for req in r["requirements"]]
    split_report = {name: {"pairs": len(rs), "cv_versions": len({r["cv_id"] for r in rs}),
                           "candidate_groups": len({r["candidate_group_id"] for r in rs}),
                           "overall_scores": dict(sorted(Counter(r["overall_score"] for r in rs).items())),
                           "queries_with_different_scores": sum(len(set(v)) > 1 for v in
                               (list(r["overall_score"] for r in rs if r["cv_id"] == cid)
                                for cid in {r["cv_id"] for r in rs}))}
                    for name, rs in splits.items()}
    train_vacancies = {r["vacancy_id"] for r in splits["train"]}
    manifest = {"schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_export": str(export_path),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "seed": seed, "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "same_candidate_overrides": list(same_candidate),
                "split_method": "candidate-group-stratified-seeded-search-v1",
                "evaluation_scope": "new candidates; vacancies may recur; five-pair annotated shortlist only",
                "review_provenance": "Saved account annotations; reviewed by Kuat and Codex at Kuat's request. The review checkbox alone is not evidence of independent human verification.",
                "label_mapping": {"0": {"status": "unknown", "relevance": None},
                                  "1": {"status": "weak", "relevance": 0},
                                  "2": {"status": "partial", "relevance": 1},
                                  "3": {"status": "good", "relevance": 2}},
                "target_policy": "Only annotations[].result. Never predictions or data.model_assessment.",
                "source_text_policy": "Original texts retained privately for traceability. Before training a text model, build professional-only inputs excluding identity/contact/demographic fields; do not feed target explanations or evidence as features.",
                "splits": split_report}
    report = {"tasks": len(rows), "cv_versions": len({r['cv_id'] for r in rows}),
              "candidate_groups": len(groups), "unique_vacancies": len({r["vacancy_id"] for r in rows}),
              "requirements": len(requirements),
              "overall_scores": dict(sorted(Counter(r["overall_score"] for r in rows).items())),
              "requirement_scores": dict(sorted(Counter(r["score"] for r in requirements).items())),
              "eligible_pairs": sum(len(rs) for rs in splits.values()),
              "quarantined_pairs": len(quarantined),
              "candidate_group_overlap": [], "cv_id_overlap": [], "validation_errors": [],
              "test_vacancies_also_in_train": len(train_vacancies & {r["vacancy_id"] for r in splits["test"]}),
              "splits": split_report}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {r["candidate_group_id"] for r in splits[a]} & {r["candidate_group_id"] for r in splits[b]}:
            raise ValueError("Candidate group leakage")
        if {r["cv_id"] for r in splits[a]} & {r["cv_id"] for r in splits[b]}:
            raise ValueError("CV ID leakage")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "pairs_all.jsonl", rows)
    write_jsonl(output / "requirements_all.jsonl", requirements)
    write_jsonl(output / "quarantine.jsonl", quarantined)
    for name, rs in splits.items():
        # Splits contain IDs and targets only. A later feature builder joins source
        # documents; this prevents accidentally training on annotation explanations.
        write_jsonl(output / f"{name}.jsonl", [{k: r[k] for k in
                    ("pair_id", "task_id", "cv_id", "candidate_group_id", "vacancy_id",
                     "overall_score", "relevance")} for r in rs])
    write_json(output / "candidate_groups.json", groups)
    write_json(output / "split_assignment.json", assignment)
    write_json(output / "manifest.json", manifest)
    write_json(output / "quality_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверить выгрузку Label Studio и подготовить датасет ML2")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/annotations/ml2_training_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--same-candidate", action="append", default=[], metavar="CV_ID,CV_ID",
                        help="Разные версии резюме одного человека; можно указать несколько групп")
    args = parser.parse_args(argv)
    report = build_dataset(args.export, args.output_dir,
                           [g.split(",") for g in args.same_candidate], args.seed,
                           args.validation_fraction, args.test_fraction)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Данные подготовлены:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}")
