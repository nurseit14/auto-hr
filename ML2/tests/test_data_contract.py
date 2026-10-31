"""Regression checks for stable document identity and exact CSV source text."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from ML2.data import load_resumes, load_vacancies, select_pairs
from ML2.label_studio import make_task


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_csv(self, name, columns, rows):
        path = self.root / name
        # Preserve embedded CRLF and tabs rather than relying on platform newline conversion.
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_document_id_takes_precedence_and_preserves_source_metadata(self):
        text = "Java Developer"
        path = self.write_csv(
            "current.csv",
            ["document_id", "resume_id", "cv_id", "text", "text_version", "source_sha256"],
            [["doc_stable", "resume_0001", "legacy_cv", text, "prep_v2", "source_fingerprint"]],
        )
        row = load_resumes(path)[0]
        self.assertEqual(row["cv_id"], "doc_stable")
        self.assertEqual(row["document_id"], "doc_stable")
        self.assertEqual(row["resume_id"], "resume_0001")
        self.assertEqual(row["text_version"], "prep_v2")
        self.assertEqual(row["source_sha256"], "source_fingerprint")

    def test_document_id_does_not_require_legacy_ids_or_optional_metadata(self):
        path = self.write_csv("minimal.csv", ["document_id", "text"], [["doc_only", "Python"]])
        row = load_resumes(path)[0]
        self.assertEqual((row["cv_id"], row["document_id"]), ("doc_only", "doc_only"))

    def test_repeated_document_id_is_rejected_despite_different_legacy_ids(self):
        path = self.write_csv(
            "duplicate.csv", ["document_id", "resume_id", "text"],
            [["same_document", "resume_0001", "Java"],
             ["same_document", "resume_0002", "Python"]],
        )
        with self.assertRaises(ValueError):
            load_resumes(path)

    def test_present_but_blank_document_id_cannot_fall_back_to_legacy(self):
        for number, blank in enumerate(("", " \t ")):
            with self.subTest(document_id=repr(blank)):
                path = self.write_csv(
                    f"blank_{number}.csv", ["document_id", "resume_id", "cv_id", "text"],
                    [[blank, "resume_valid", "cv_valid", "Java"]],
                )
                with self.assertRaises(ValueError):
                    load_resumes(path)

    def test_legacy_csv_uses_resume_id_then_cv_id_then_exact_text_hash(self):
        text = "  Қазақша Python\r\n\t"
        path = self.write_csv(
            "legacy.csv", ["resume_id", "cv_id", "text"],
            [["resume_first", "cv_ignored", "Java"],
             ["", "cv_second", "SQL"], ["", "", text]],
        )
        rows = load_resumes(path)
        self.assertEqual([r["cv_id"] for r in rows],
                         ["resume_first", "cv_second", "cv_" + text_hash(text)[:16]])
        self.assertEqual(rows[-1]["text"], text)

    def test_resume_text_and_hash_preserve_spaces_crlf_tabs_and_unicode(self):
        text = " \tҚуат — Java әзірлеуші 🧪\r\nSkills:\tJava, SQL\r\n  "
        path = self.write_csv("exact.csv", ["document_id", "text"], [["doc_exact", text]])
        row = load_resumes(path)[0]
        self.assertEqual(row["text"], text)
        self.assertEqual(row["text_sha256"], text_hash(text))

    def test_matching_incoming_text_hash_is_accepted(self):
        text = "\r\n Java\t "
        path = self.write_csv(
            "matching_hash.csv", ["document_id", "text", "text_sha256"],
            [["doc_hash", text, text_hash(text)]],
        )
        self.assertEqual(load_resumes(path)[0]["text_sha256"], text_hash(text))

    def test_nonempty_incoming_hash_mismatch_is_rejected(self):
        text = "  Java\r\n"
        path = self.write_csv(
            "stale_hash.csv", ["document_id", "text", "text_sha256"],
            [["doc_hash", text, text_hash(text.strip())]],
        )
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            load_resumes(path)
        self.assertEqual(path.read_bytes(), original)

    def test_present_but_blank_incoming_hash_is_rejected(self):
        text = "\tPython\r\n "
        for number, blank in enumerate(("", " \t\r\n ")):
            with self.subTest(text_sha256=repr(blank)):
                path = self.write_csv(
                    f"empty_hash_{number}.csv", ["document_id", "text", "text_sha256"],
                    [["doc_hash", text, blank]],
                )
                with self.assertRaises(ValueError):
                    load_resumes(path)

    def test_whitespace_only_vacancy_has_no_source_text(self):
        path = self.write_csv(
            "empty_job.csv",
            ["vacancy_id", "title", "experience", "requirements", "responsibilities", "text"],
            [["vac_empty", " \t", "\r\n", "\t ", "  ", " \r\n\t "]],
        )
        with self.assertRaisesRegex(ValueError, "no source text"):
            load_vacancies(path)

    def test_vacancy_source_fields_and_hash_retain_exact_whitespace(self):
        title = "  Java Developer\t"
        requirements = "\tJava\r\nSQL  "
        source = " \r\nҚазақша мәтін 🧪\t "
        path = self.write_csv(
            "vacancies.csv", ["vacancy_id", "title", "requirements", "text"],
            [["vac_exact", title, requirements, source]],
        )
        vacancy = load_vacancies(path)[0]
        self.assertEqual(vacancy["title"], title)
        self.assertEqual(vacancy["requirements"], requirements)
        self.assertEqual(vacancy["text"], source)
        expected = (f"Название:\n{title}\n\nТребования:\n{requirements}"
                    f"\n\nСохранённый текст:\n{source}")
        self.assertEqual(vacancy["display"], expected)
        self.assertEqual(vacancy["source_sha256"], text_hash(expected))

    def test_json_skills_surrounded_by_whitespace_parse_for_both_sources(self):
        encoded = ' \t["Java", "SQL"]\r\n '
        cvs = self.write_csv("skills_cv.csv", ["resume_id", "text", "skills"],
                             [["resume_skills", "Java SQL", encoded]])
        jobs = self.write_csv("skills_job.csv", ["vacancy_id", "text", "skills"],
                              [["vac_skills", "Java SQL", encoded]])
        self.assertEqual(load_resumes(cvs)[0]["skills"], ["java", "sql"])
        self.assertEqual(load_vacancies(jobs)[0]["skills"], ["java", "sql"])

    def test_pair_and_label_studio_identity_and_offsets_survive_renumbering(self):
        cv_text = "\r\n  Java\tәзірлеуші\r\n "
        requirement = "\tJava\r\nSQL  "
        job_source = "  Java\tSQL\r\n "
        jobs_path = self.write_csv(
            "jobs.csv", ["vacancy_id", "title", "requirements", "text"],
            [["vac_stable", "Java Developer", requirement, job_source]],
        )
        vacancies = load_vacancies(jobs_path)
        pairs = []
        for number, resume_id in enumerate(("resume_0001", "resume_0099")):
            path = self.write_csv(
                f"renumbered_{number}.csv", ["document_id", "resume_id", "text", "text_sha256"],
                [["doc_stable", resume_id, cv_text, text_hash(cv_text)]],
            )
            pairs.append(select_pairs(load_resumes(path), vacancies, count=1)[0])
        self.assertEqual(pairs[0]["pair_id"], pairs[1]["pair_id"])
        record = {"model_version": "fictional_test", "assessment": {
            "warnings": [], "overall_score": 2, "overall_explanation": "Проверка формата.",
            "requirements": [{"category": "HARD_SKILL", "requirement": "Java и SQL",
                              "vacancy_quote": requirement, "score": 2,
                              "explanation": "Java указана в исходном тексте.",
                              "cv_quote": "Java\tәзірлеуші"}],
        }}
        task = make_task(pairs[1], record, "fictional_dataset")
        data = task["data"]
        self.assertEqual(data["cv_id"], "doc_stable")
        self.assertEqual(data["cv_text"], cv_text)
        self.assertEqual(data["cv_text_sha256"], text_hash(cv_text))
        self.assertEqual(data["vacancy_fields"]["text"], job_source)
        self.assertEqual(data["vacancy_text_sha256"], text_hash(data["vacancy_display"]))
        results = task["predictions"][0]["result"]
        span = next(r["value"] for r in results if r["type"] == "labels")
        self.assertEqual(span["text"], requirement)
        self.assertEqual(data["vacancy_display"][span["start"]:span["end"]], requirement)
        evidence = next(r["value"]["text"][0] for r in results if r["from_name"] == "cv_evidence")
        self.assertEqual(evidence, "Java\tәзірлеуші")
        self.assertIn(evidence, data["cv_text"])


if __name__ == "__main__":
    unittest.main()
