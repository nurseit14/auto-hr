import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from NLP import process_resumes


class ProcessingErrorTests(unittest.TestCase):

    def test_failed_page_is_visible_to_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            output_file = output_dir / "resumes.csv"
            input_dir.mkdir()

            pdf = input_dir / "synthetic.pdf"
            doc = pymupdf.open()
            doc.new_page().insert_text(
                (50, 80), "SYNTHETIC TEST DOCUMENT"
            )
            doc.save(pdf)
            doc.close()

            fake_result = {
                "raw_text": "Partial CV text",
                "clean_text": "Partial CV text",
                "text": "Partial CV text",
                "skills": [],
                "skill_count": 0,
                "review_required": True,
                "layouts": [
                    {
                        "page": 1,
                        "layout": "failed",
                        "error": "Synthetic extraction failure",
                    }
                ],
            }

            output = io.StringIO()

            with (
                patch.object(
                    process_resumes, "RESUMES_DIR", input_dir
                ),
                patch.object(
                    process_resumes, "OUTPUT_DIR", output_dir
                ),
                patch.object(
                    process_resumes, "OUTPUT_FILE", output_file
                ),
                patch.object(
                    process_resumes,
                    "parse_cv",
                    return_value=fake_result,
                ),
                contextlib.redirect_stdout(output),
            ):
                process_resumes.main()

            with output_file.open(
                encoding="utf-8-sig", newline=""
            ) as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            errors = json.loads(row["extraction_errors"])

            self.assertEqual(row["review_required"], "True")
            self.assertEqual(row["layouts"], "p1=failed")
            self.assertEqual(
                errors,
                [{
                    "page": 1,
                    "error": "Synthetic extraction failure",
                }],
            )

            terminal_output = output.getvalue()
            self.assertIn("WARNING: page 1", terminal_output)
            self.assertIn(
                "Synthetic extraction failure",
                terminal_output,
            )
            self.assertIn(
                "ACTION: manual review required",
                terminal_output,
            )


if __name__ == "__main__":
    unittest.main()
