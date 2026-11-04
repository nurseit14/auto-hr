import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

from ML2.blind_review import (ASSETS, TASK_FIELDS, assert_blind_tasks, build_packet,
                             read_dataset, render_html, select_rows, sha256,
                             source_snapshot, packet_identity)


def make_dataset(directory):
    """Fictional pilot with rare positives, one empty CV and one repeated person."""
    directory.mkdir()
    rows, split_rows = [], {"train": [], "validation": [], "test": []}
    for cv in range(44):
        group = f"candidate_{9 if cv == 10 else cv}"
        cv_text = "" if cv == 43 else f"Fictional CV {cv}\r\nBackend developer. Python and Django."
        for job in range(5):
            pair_id = f"ORIGINAL_PAIR_CANARY_{cv}_{job}"
            score = 0 if cv == 43 else 3 if cv < 5 and job < 2 else 1 + (cv + job) % 2
            vacancy = f"Fictional Python Backend Developer {job}\nPython and Django."
            row = {"pair_id": pair_id, "task_id": cv * 5 + job,
                   "cv_id": f"ORIGINAL_CV_CANARY_{cv}", "candidate_group_id": group,
                   "vacancy_id": f"ORIGINAL_JOB_CANARY_{job}",
                   "cv_text": cv_text, "vacancy_display": vacancy,
                   "vacancy_title": f"Backend Developer {job}",
                   "cv_text_sha256": sha256(cv_text), "vacancy_text_sha256": sha256(vacancy),
                   "overall_score": score, "relevance": score - 1 if score else None,
                   "overall_explanation": "OLD_OVERALL_EXPLANATION_CANARY",
                   "annotation_source": {"annotation_id": "OLD_ANNOTATION_CANARY"},
                   "requirements": [{"score": score, "category": "EXPERIENCE",
                                     "cv_quote": "OLD_EVIDENCE_CANARY",
                                     "explanation": "OLD_REQUIREMENT_EXPLANATION_CANARY"}],
                   "predictions": [{"score": "MODEL_PREDICTION_CANARY"}],
                   "selection": {"selection_score": 99.77, "selection_rank": job + 1}}
            rows.append(row)
            if cv != 43:
                split = "train" if cv < 31 else "validation" if cv < 37 else "test"
                split_rows[split].append({key: row[key] for key in
                                         ("pair_id", "candidate_group_id", "cv_id", "vacancy_id")})
    (directory / "pairs_all.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"source_export": "not-present.json",
                                                        "source_sha256": "unavailable-legacy-export"}), encoding="utf-8")
    for split, items in split_rows.items():
        (directory / f"{split}.jsonl").write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")
    return rows


class BlindPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "source"
        self.rows = make_dataset(self.dataset)

    def tearDown(self):
        self.temp.cleanup()

    def test_fifty_unique_pairs_and_strata_with_group_cap(self):
        rows, _, splits = read_dataset(self.dataset)
        chosen, quotas, cap = select_rows(rows, 50, 42, splits)
        self.assertEqual(len(chosen), 50)
        self.assertEqual(len({row["pair_id"] for row in chosen}), 50)
        self.assertEqual(Counter(row["overall_score"] for row in chosen), {0: 1, 1: 21, 2: 21, 3: 7})
        self.assertEqual(quotas, {0: 1, 1: 21, 2: 21, 3: 7})
        self.assertEqual(cap, 2)
        self.assertLessEqual(max(Counter(row["candidate_group_id"] for row in chosen).values()), 2)
        self.assertGreaterEqual(len({row["candidate_group_id"] for row in chosen}), 40)

    def test_reviewer_payload_has_no_original_ids_or_judgments(self):
        result = build_packet(self.dataset, self.root / "packet", reference_date="2026-10-21")
        reviewer = Path(result["reviewer_dir"])
        payload = json.loads((reviewer / "tasks.json").read_text())
        assert_blind_tasks(payload)
        for task in payload:
            self.assertEqual(set(task), {"data"})
            self.assertEqual(set(task["data"]), TASK_FIELDS)
            self.assertEqual(task["data"]["reference_date"], "2026-10-21")
        serialized = (reviewer / "tasks.json").read_text()
        for secret in ("ORIGINAL_PAIR_CANARY", "ORIGINAL_CV_CANARY", "ORIGINAL_JOB_CANARY",
                       "OLD_OVERALL_EXPLANATION_CANARY", "OLD_ANNOTATION_CANARY",
                       "OLD_EVIDENCE_CANARY", "OLD_REQUIREMENT_EXPLANATION_CANARY",
                       "MODEL_PREDICTION_CANARY", "selection_score", "overall_score", "predictions"):
            self.assertNotIn(secret, serialized)
        mapping = json.loads((Path(result["coordinator_dir"]) / "pair_mapping_PRIVATE.json").read_text())
        self.assertEqual(len(mapping["pairs"]), 50)
        by_review = {row["review_id"]: row for row in mapping["pairs"]}
        original = {row["pair_id"]: row for row in self.rows}
        for task in payload:
            data = task["data"]
            prior = original[by_review[data["review_id"]]["pair_id"]]
            self.assertEqual(data["cv_text"], prior["cv_text"])
            self.assertEqual(data["vacancy_display"], prior["vacancy_display"])
        self.assertTrue((reviewer / "blind_label_config.xml").is_file())
        self.assertTrue((reviewer / "ANNOTATION_GUIDELINES.md").is_file())

    def test_generation_is_reproducible_and_source_is_unchanged(self):
        before = source_snapshot(self.dataset)
        first = build_packet(self.dataset, self.root / "first")
        second = build_packet(self.dataset, self.root / "second")
        self.assertEqual(first["tasks"], second["tasks"])
        self.assertEqual(first["packet_id"], second["packet_id"])
        self.assertEqual(before, source_snapshot(self.dataset))
        source_order = [row["pair_id"] for row in self.rows]
        mapping = json.loads((Path(first["coordinator_dir"]) / "pair_mapping_PRIVATE.json").read_text())
        order = [source_order.index(row["pair_id"]) for row in mapping["pairs"]]
        self.assertNotEqual(order, sorted(order))

    def test_unsafe_outputs_and_overwrites_are_rejected(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
        for path in (output, self.dataset, self.dataset / "new-packet", self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                build_packet(self.dataset, path)
        self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_too_many_pairs_or_invalid_reference_date_are_rejected(self):
        for count in (0, -1, 221):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_packet(self.dataset, self.root / f"out{count}", count=count)
        with self.assertRaises(ValueError):
            build_packet(self.dataset, self.root / "bad-date", reference_date="2026-99-99")

    def test_tampered_source_hash_is_rejected_without_creating_packet(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["cv_text"] += "modified"
        (self.dataset / "pairs_all.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_packet(self.dataset, self.root / "packet")
        self.assertFalse((self.root / "packet").exists())

    def test_source_split_group_leakage_is_rejected(self):
        row = {key: self.rows[1][key] for key in ("pair_id", "candidate_group_id", "cv_id", "vacancy_id")}
        # Keep unique pair assignment but move a member of the same group.
        train = [json.loads(line) for line in (self.dataset / "train.jsonl").read_text().splitlines()]
        (self.dataset / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train if r["pair_id"] != row["pair_id"]))
        with (self.dataset / "validation.jsonl").open("a") as stream:
            stream.write("\n" + json.dumps(row))
        with self.assertRaisesRegex(ValueError, "group leakage"):
            build_packet(self.dataset, self.root / "packet")

    def test_blind_allowlist_rejects_nested_targets(self):
        task = {"data": {"review_id": "BR-0123456789abcdef", "cv_text": "CV",
                         "vacancy_display": "Job", "reference_date": "2026-10-21",
                         "overall_score": 3}}
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            assert_blind_tasks([task])
        del task["data"]["overall_score"]
        task["annotations"] = []
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            assert_blind_tasks([task])

    def test_offline_json_is_safe_against_source_markup(self):
        tasks = [{"data": {"review_id": "BR-0123456789abcdef", "cv_text": '</script><img src=x onerror=alert(1)>\u2028&__AUTOHR_BLIND_META_JSON____AUTOHR_BLIND_TASKS_JSON__',
                           "vacancy_display": "Job", "reference_date": "2026-10-21"}}]
        template = '<script type="application/json">__AUTOHR_BLIND_TASKS_JSON__</script><script>__AUTOHR_BLIND_META_JSON__</script>'
        html = render_html(tasks, "blind-test", template)
        self.assertNotIn("<img", html)
        self.assertNotIn("\u2028", html)
        encoded = html.split('application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(encoded), tasks)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            render_html(tasks, "test", template + "__AUTOHR_BLIND_TASKS_JSON__")

    def test_packet_identity_binds_selection_order_policy_and_rubric(self):
        original = packet_identity("sourcehash", ["pair-a", "pair-b"], 42, "2026-10-21")
        self.assertNotEqual(original, packet_identity("sourcehash", ["pair-a", "pair-c"], 42, "2026-10-21"))
        self.assertNotEqual(original, packet_identity("sourcehash", ["pair-b", "pair-a"], 42, "2026-10-21"))
        self.assertNotEqual(original, packet_identity("sourcehash", ["pair-a", "pair-b"], 42, "2026-10-21", policy="new-policy"))
        self.assertNotEqual(original, packet_identity("sourcehash", ["pair-a", "pair-b"], 42, "2026-10-21", rubric_version="2.0"))

    def test_label_studio_form_has_blank_controls_and_reviewer_identity(self):
        root = ElementTree.fromstring((ASSETS / "blind_label_config.xml").read_text())
        names = {node.attrib.get("name") for node in root.iter()}
        self.assertIn("reviewer_id", names)
        self.assertIn("human_review", names)
        for node in root.iter():
            self.assertNotIn("selected", node.attrib)
            self.assertNotIn("defaultValue", node.attrib)
        values = [node.attrib.get("value", "") for node in root.iter()]
        for forbidden in ("$overall_score", "$selection_rank", "$selection_score", "$model_assessment", "$cv_id", "$pair_id"):
            self.assertFalse(any(forbidden in value for value in values))


if __name__ == "__main__":
    unittest.main()
