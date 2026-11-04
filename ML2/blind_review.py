"""Build a private coordinator record and a strictly blinded reviewer packet.

The packet samples saved judgments for quality review; it never changes the
frozen dataset or uses the second participant's future answers for training.
Only output/reviewer is intended for the second participant.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil


VERSION = "autohr-blind-review-packet-v1"
RUBRIC_VERSION = "1.0"
SELECTION_POLICY = "purposeful-label-strata-multistart-group-profile-job-risk-coverage-v1"
TASK_FIELDS = frozenset({"review_id", "cv_text", "vacancy_display", "reference_date"})
FROZEN_FILES = ("pairs_all.jsonl", "manifest.json", "quality_report.json",
                "candidate_groups.json", "split_assignment.json", "train.jsonl",
                "validation.jsonl", "test.jsonl", "quarantine.jsonl")
ASSETS = Path(__file__).resolve().parent


def sha256(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def source_snapshot(dataset):
    dataset = Path(dataset).resolve()
    return {name: sha256((dataset / name).read_bytes()) for name in FROZEN_FILES
            if (dataset / name).is_file()}


def read_dataset(dataset):
    dataset = Path(dataset).resolve()
    rows = read_jsonl(dataset / "pairs_all.jsonl")
    if not rows:
        raise ValueError("The source pair dataset is empty")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    pair_ids, pair_keys = set(), set()
    source_by_pair = {}
    cv_hashes, vacancy_hashes = {}, {}
    for row in rows:
        for key in ("pair_id", "cv_id", "vacancy_id", "candidate_group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing source identifier: {key}")
        if row["pair_id"] in pair_ids:
            raise ValueError("Duplicate source pair ID")
        pair_ids.add(row["pair_id"])
        source_by_pair[row["pair_id"]] = row
        pair_key = (row["cv_id"], row["vacancy_id"])
        if pair_key in pair_keys:
            raise ValueError("Duplicate source CV-vacancy pair")
        pair_keys.add(pair_key)
        if type(row.get("overall_score")) is not int or row["overall_score"] not in range(4):
            raise ValueError("Source overall score must be 0, 1, 2 or 3")
        for text_key, hash_key, identity_key, seen in (
                ("cv_text", "cv_text_sha256", "cv_id", cv_hashes),
                ("vacancy_display", "vacancy_text_sha256", "vacancy_id", vacancy_hashes)):
            text = row.get(text_key)
            if not isinstance(text, str) or row.get(hash_key) != sha256(text):
                raise ValueError(f"Source text hash mismatch: {text_key}")
            identity = row[identity_key]
            if identity in seen and seen[identity] != row[hash_key]:
                raise ValueError(f"Conflicting source text versions: {identity_key}")
            seen[identity] = row[hash_key]
        if not row["vacancy_display"].strip():
            raise ValueError("Source vacancy text is empty")
        if not row["cv_text"].strip() and row["overall_score"] != 0:
            raise ValueError("An empty source CV must be unknown")
    split_by_pair = {}
    group_splits = {}
    for split in ("train", "validation", "test"):
        path = dataset / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row["pair_id"] not in pair_ids:
                raise ValueError("Split contains a pair missing from the source")
            if row["pair_id"] in split_by_pair:
                raise ValueError("A source pair appears in multiple splits")
            original = source_by_pair[row["pair_id"]]
            for field in ("candidate_group_id", "cv_id", "vacancy_id", "overall_score", "relevance"):
                if field in row and row[field] != original.get(field):
                    raise ValueError("Frozen split metadata differs from its source pair")
            split_by_pair[row["pair_id"]] = split
            group = row["candidate_group_id"]
            if group in group_splits and group_splits[group] != split:
                raise ValueError("Candidate group leakage between frozen splits")
            group_splits[group] = split
    # The legacy source export is optional when a frozen dataset is copied to
    # another machine. If available, verify the recorded origin without editing it.
    export = Path(manifest.get("source_export", ""))
    if export.is_file() and manifest.get("source_sha256"):
        if sha256(export.read_bytes()) != manifest["source_sha256"]:
            raise ValueError("The source Label Studio export differs from its manifest")
    return rows, manifest, split_by_pair


def profile(row):
    title = row.get("vacancy_title", "").casefold()
    families = (
        ("data-engineering", r"data.?engineer|инженер.*данн|etl|dwh"),
        ("data-science", r"data.?scien|machine.?learning|\bml\b|искусствен"),
        ("data-analysis", r"data.?anal|аналитик|business.?intelligence|\bbi\b"),
        ("fullstack", r"full.?stack|фулст"),
        ("frontend", r"front|react|angular|vue|версталь"),
        ("quality", r"\bqa\b|тестир|quality|test.?engineer"),
        ("infrastructure", r"devops|sre|системн|админ|network|cloud|сетев"),
        ("security", r"security|безопас|кибер"),
        ("mobile", r"mobile|android|ios|мобиль"),
        ("design", r"design|дизайн|ux|ui"),
        ("backend", r"back|python|java|php|c#|\.net|разработ|developer|programmer"),
    )
    return next((name for name, pattern in families if re.search(pattern, title)), "other-it")


def risk_tags(row):
    """Coordinator-only coverage hints from existing judgments, not new labels."""
    requirements = row.get("requirements", [])
    tags = set()
    statuses = {r.get("score") for r in requirements}
    if 0 in statuses and statuses.intersection({1, 2, 3}):
        tags.add("mixed-known-unknown")
    if any(r.get("category") in {"ROLE_LEVEL", "EXPERIENCE"} and r.get("score") in {1, 2}
           for r in requirements):
        tags.add("role-or-experience-shortfall")
    if 2 in statuses:
        tags.add("partial-requirement")
    if not row["cv_text"].strip():
        tags.add("empty-cv")
    return sorted(tags)


def label_quotas(rows, count):
    """Purposeful quality-audit strata; not a population-proportional sample."""
    available = Counter(row["overall_score"] for row in rows)
    weights = {0: 1, 1: 21, 2: 21, 3: 7}
    quotas = {label: min(available[label], int(count * weight / 50))
              for label, weight in weights.items()}
    fractions = {label: count * weights[label] / 50 - int(count * weights[label] / 50)
                 for label in weights}
    while sum(quotas.values()) < count:
        eligible = [label for label in weights if quotas[label] < available[label]]
        if not eligible:
            raise ValueError("Not enough source pairs for the requested count")
        label = max(eligible, key=lambda value: (fractions[value],
                    count * weights[value] / 50 - quotas[value], available[value] - quotas[value], -value))
        quotas[label] += 1
        fractions[label] = -1
    return quotas


def select_rows(rows, count=50, seed=42, splits=None):
    if type(count) is not int or count < 1 or count > len(rows):
        raise ValueError("Requested count must be positive and no larger than the source dataset")
    splits = splits or {}
    rows = sorted(rows, key=lambda row: row["pair_id"])
    quotas = label_quotas(rows, count)
    group_count = len({row["candidate_group_id"] for row in rows})
    # With a small fixture or unusual requested count a cap of two is infeasible.
    # Any relaxed cap is explicit in the private audit, never silently assumed.
    minimum_cap = max(2, (count + group_count - 1) // group_count)
    best, best_objective, chosen_cap = None, None, None
    for cap in range(minimum_cap, max(minimum_cap, count) + 1):
        for attempt in range(96):
            rng = random.Random(f"{seed}:selection:{cap}:{attempt}")
            jitter = {row["pair_id"]: rng.random() for row in rows}
            selected, remaining = [], list(rows)
            labels, groups, cvs, profiles, jobs, split_counts, risks = (Counter() for _ in range(7))
            while len(selected) < count:
                available = [row for row in remaining
                             if labels[row["overall_score"]] < quotas[row["overall_score"]]
                             and groups[row["candidate_group_id"]] < cap]
                if not available:
                    break
                def priority(row):
                    group = row["candidate_group_id"]
                    family = profile(row)
                    split = splits.get(row["pair_id"], "quarantine")
                    # Cover people first. Then diversify roles, jobs and risk
                    # patterns, while including the source split's smaller slices.
                    return (groups[group] == 0, cvs[row["cv_id"]] == 0,
                            profiles[family] == 0, jobs[row["vacancy_id"]] == 0,
                            sum(risks[tag] == 0 for tag in risk_tags(row)),
                            -split_counts[split], -groups[group], jitter[row["pair_id"]])
                picked = max(available, key=priority)
                selected.append(picked)
                remaining.remove(picked)
                labels[picked["overall_score"]] += 1
                groups[picked["candidate_group_id"]] += 1
                cvs[picked["cv_id"]] += 1
                profiles[profile(picked)] += 1
                jobs[picked["vacancy_id"]] += 1
                split_counts[splits.get(picked["pair_id"], "quarantine")] += 1
                risks.update(risk_tags(picked))
            if len(selected) != count:
                continue
            objective = (len(groups), len(cvs), len(profiles), len(jobs), len(risks),
                         min(split_counts.get("validation", 0), split_counts.get("test", 0)))
            if best_objective is None or objective > best_objective:
                best, best_objective, chosen_cap = selected, objective, cap
        if best is not None:
            break
    if best is None:
        raise ValueError("Could not satisfy the selection quotas")
    return best, quotas, chosen_cap


def assert_blind_tasks(tasks):
    seen = set()
    for task in tasks:
        if set(task) != {"data"} or set(task["data"]) != TASK_FIELDS:
            raise ValueError("Reviewer tasks contain non-allowlisted fields")
        data = task["data"]
        if not re.fullmatch(r"BR-[0-9a-f]{16}", data["review_id"]):
            raise ValueError("Invalid opaque review ID")
        if data["review_id"] in seen:
            raise ValueError("Duplicate blind review ID")
        seen.add(data["review_id"])
        if not all(isinstance(data[key], str) for key in TASK_FIELDS):
            raise ValueError("Reviewer task values must be strings")
        date.fromisoformat(data["reference_date"])
    if not tasks:
        raise ValueError("Reviewer tasks are empty")


def script_json(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                     ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(old, new)
    return text


def render_html(tasks, packet_id, template):
    assert_blind_tasks(tasks)
    markers = {"__AUTOHR_BLIND_TASKS_JSON__": tasks,
               "__AUTOHR_BLIND_META_JSON__": {"packet_id": packet_id, "rubric_version": RUBRIC_VERSION}}
    for marker in markers:
        if template.count(marker) != 1:
            raise ValueError(f"HTML template must contain exactly one {marker}")
    # One pass over the ORIGINAL template: marker-looking source text must never
    # be processed as a template token after it has been embedded in JSON.
    pattern = re.compile("|".join(re.escape(marker) for marker in markers))
    return pattern.sub(lambda match: script_json(markers[match.group(0)]), template)


def packet_identity(source_hash, ordered_pair_ids, seed, reference_date,
                    policy=SELECTION_POLICY, rubric_version=RUBRIC_VERSION):
    identity = {"schema_version": VERSION, "source_sha256": source_hash,
                "ordered_selected_pair_ids": ordered_pair_ids, "seed": seed,
                "reference_date": reference_date, "selection_policy": policy,
                "rubric_version": rubric_version}
    return "blind_" + sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")))[:20]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_packet(dataset, output_dir, count=50, seed=42, reference_date="2026-10-21"):
    dataset, output_dir = Path(dataset).resolve(), Path(output_dir).resolve()
    date.fromisoformat(reference_date)
    if output_dir == dataset or dataset in output_dir.parents or output_dir in dataset.parents:
        raise ValueError("Packet output must be separate from the frozen dataset")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory is not empty; choose a new packet version")
    before = source_snapshot(dataset)
    rows, source_manifest, splits = read_dataset(dataset)
    selected, quotas, cap = select_rows(rows, count, seed, splits)
    source_hash = before["pairs_all.jsonl"]
    shuffled = list(selected)
    random.Random(f"{seed}:presentation:{source_hash}").shuffle(shuffled)
    packet_id = packet_identity(source_hash, [row["pair_id"] for row in shuffled], seed, reference_date)
    tasks, mapping = [], []
    for row in shuffled:
        review_id = "BR-" + sha256(f"{packet_id}:{row['pair_id']}:blind-alias")[:16]
        tasks.append({"data": {"review_id": review_id, "cv_text": row["cv_text"],
                               "vacancy_display": row["vacancy_display"],
                               "reference_date": reference_date}})
        mapping.append({"review_id": review_id, "pair_id": row["pair_id"],
                        "task_id": row.get("task_id"), "cv_id": row["cv_id"],
                        "vacancy_id": row["vacancy_id"],
                        "candidate_group_id": row["candidate_group_id"],
                        "split": splits.get(row["pair_id"], "quarantine"),
                        "original_overall_score": row["overall_score"],
                        "original_relevance": row.get("relevance"),
                        "original_overall_explanation": row.get("overall_explanation", ""),
                        "original_requirements": row.get("requirements", []),
                        "annotation_source": row.get("annotation_source", {}),
                        "cv_text_sha256": row["cv_text_sha256"],
                        "vacancy_text_sha256": row["vacancy_text_sha256"],
                        "selection_profile_hint": profile(row),
                        "selection_risk_hints": risk_tags(row)})
    assert_blind_tasks(tasks)
    chosen_ids = {row["pair_id"] for row in selected}
    summary = {"source_pairs": len(rows), "selected_pairs": len(selected),
               "candidate_groups": len({row["candidate_group_id"] for row in selected}),
               "cv_versions": len({row["cv_id"] for row in selected}),
               "unique_vacancies": len({row["vacancy_id"] for row in selected}),
               "old_label_counts": dict(sorted(Counter(row["overall_score"] for row in selected).items())),
               "source_split_counts": dict(sorted(Counter(splits.get(row["pair_id"], "quarantine")
                                                            for row in selected).items())),
               "profile_hints": dict(sorted(Counter(profile(row) for row in selected).items())),
               "risk_hints": dict(sorted(Counter(tag for row in selected for tag in risk_tags(row)).items())),
               "maximum_pairs_per_candidate_group": max(Counter(row["candidate_group_id"] for row in selected).values()),
               "group_cap": cap}
    after = source_snapshot(dataset)
    if before != after:
        raise ValueError("The source dataset changed during packet preparation")
    # Assets are repository files, never taken from the annotated source export.
    template_path = ASSETS / "blind_review_template.html"
    html = render_html(tasks, packet_id, template_path.read_text(encoding="utf-8")) if template_path.exists() else None
    config = (ASSETS / "blind_label_config.xml").read_text(encoding="utf-8")
    reviewer, coordinator = output_dir / "reviewer", output_dir / "coordinator"
    reviewer.mkdir(parents=True)
    coordinator.mkdir()
    write_json(reviewer / "tasks.json", tasks)
    (reviewer / "blind_label_config.xml").write_text(config, encoding="utf-8")
    for source, target in (("ANNOTATION_GUIDELINES.md", "ANNOTATION_GUIDELINES.md"),
                           ("BLIND_REVIEW_INSTRUCTIONS.md", "README.md")):
        if (ASSETS / source).is_file():
            shutil.copyfile(ASSETS / source, reviewer / target)
    if html is not None:
        (reviewer / "START_REVIEW.html").write_text(html, encoding="utf-8")
    write_json(coordinator / "pair_mapping_PRIVATE.json", {"packet_id": packet_id, "pairs": mapping})
    write_json(coordinator / "selection_manifest_PRIVATE.json", {
        "schema_version": VERSION, "packet_id": packet_id, "rubric_version": RUBRIC_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_dataset": str(dataset),
        "reference_date": reference_date, "seed": seed, "source_fingerprints": before,
        "source_export_sha256": source_manifest.get("source_sha256"),
        "selection_policy": SELECTION_POLICY,
        "selection_note": "Old labels select audit strata privately; reviewers see none. This enriched sample is not population-proportional and cannot estimate dataset-wide agreement without weighting or a separate representative sample.",
        "quotas": quotas, "summary": summary,
        "excluded_pairs": [{"pair_id": row["pair_id"], "reason": "not_selected_in_fixed_size_quality_audit"}
                           for row in rows if row["pair_id"] not in chosen_ids],
        "reviewer_payload_policy": sorted(TASK_FIELDS),
        "source_files_unchanged": before == after,
        "historical_splits_unchanged": True,
        "future_review_policy": "Keep independent answers separate. Do not tune on historical test reviews or overwrite frozen labels; adjudicate into a new version."})
    return {"packet_id": packet_id, "reviewer_dir": str(reviewer),
            "coordinator_dir": str(coordinator), "tasks": tasks, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a blind second-participant review packet")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-date", default="2026-10-21",
                        help="Frozen source assessment date, not the packet creation date")
    args = parser.parse_args(argv)
    result = build_packet(args.dataset, args.output_dir, args.count, args.seed, args.reference_date)
    print(json.dumps({key: result[key] for key in ("packet_id", "reviewer_dir", "coordinator_dir")}, ensure_ascii=False, indent=2))
    print(f"Prepared {len(result['tasks'])} blind tasks. Share only the reviewer directory.")


if __name__ == "__main__":
    main()
