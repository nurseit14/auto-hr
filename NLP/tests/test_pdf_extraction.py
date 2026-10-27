import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from NLP import cv_parser


class PDFExtractionTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "test.pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 80), "TEST PDF PAGE")
        doc.save(self.pdf_path)
        doc.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_column_reading_order(self):
        self.assertTrue(
            cv_parser.run_column_regression_test(),
            "Column order regression failed",
        )

    def test_cleaning_regressions(self):
        self.assertTrue(
            cv_parser.run_cleaning_regression_test(),
            "Text cleaning regression failed",
        )

    def test_anonymization_preserves_dates(self):
        from NLP.anonymizer import anonymize_text

        original = (
            "Email: test@example.com\\n"
            "Phone: +7 777 123 45 67\\n"
            "Phone: 8 701 222 33 44\\n"
            "Employment: 2019 - 2023\\n"
            "Employment: 09.2022 - 06.2025\\n"
            "Education: 2020 - 2024\\n"
        )

        result = anonymize_text(original)

        self.assertIn("[EMAIL]", result)
        self.assertEqual(result.count("[PHONE]"), 2)
        self.assertNotIn("test@example.com", result)
        self.assertNotIn("+7 777 123 45 67", result)

        self.assertIn("2019 - 2023", result)
        self.assertIn("09.2022 - 06.2025", result)
        self.assertIn("2020 - 2024", result)

    def test_oversized_header_keeps_reading_order(self):
        from NLP.cv_parser import (
            classify_two_column_blocks,
            order_two_column_blocks,
        )

        blocks = [
            (226.8, 37.7, 368.7, 203.2, "AB", 0, 0),
            (40.0, 79.8, 449.6, 115.9, "COMMON HEADER", 1, 0),
            (40.0, 220.0, 170.0, 235.0, "CONTACT", 2, 0),
            (40.0, 245.0, 170.0, 260.0, "EDUCATION", 3, 0),
            (300.0, 220.0, 480.0, 235.0, "SUMMARY", 4, 0),
            (300.0, 245.0, 480.0, 260.0, "EXPERIENCE", 5, 0),
        ]

        gutter = (180, 280)
        groups = classify_two_column_blocks(blocks, gutter)

        self.assertIsNotNone(groups)

        header, lead, left, right = groups

        self.assertIn("AB", [b[4] for b in header])
        self.assertNotIn("AB", [b[4] for b in right])

        text = order_two_column_blocks(blocks, gutter)

        self.assertLess(text.index("AB"), text.index("CONTACT"))
        self.assertLess(text.index("COMMON HEADER"), text.index("SUMMARY"))

    def test_normal_pdf(self):
        text, layouts = cv_parser.extract_pdf_with_layout(
            self.pdf_path
        )
        self.assertIn("TEST PDF PAGE", text)
        self.assertEqual(layouts[0]["layout"], "single")

    def test_page_extraction_error(self):
        with patch.object(
            cv_parser,
            "extract_page_text",
            side_effect=RuntimeError("Synthetic page error"),
        ):
            _, layouts = cv_parser.extract_pdf_with_layout(
                self.pdf_path
            )

        self.assertEqual(layouts[0]["layout"], "failed")
        self.assertIn("Synthetic page error", layouts[0]["error"])

    def test_empty_ocr(self):
        with (
            patch.object(
                cv_parser,
                "extract_page_text",
                return_value=("", "single"),
            ),
            patch.object(
                cv_parser,
                "ocr_page",
                return_value="",
            ),
        ):
            _, layouts = cv_parser.extract_pdf_with_layout(
                self.pdf_path
            )

        self.assertEqual(layouts[0]["layout"], "failed")
        self.assertEqual(
            layouts[0]["error"],
            "OCR returned empty text",
        )

    def test_failed_page_requires_review(self):
        with patch.object(
            cv_parser,
            "extract_pdf_with_layout",
            return_value=(
                "Test CV content",
                [{"page": 1, "layout": "failed", "error": "test"}],
            ),
        ):
            result = cv_parser.parse_cv(self.pdf_path)

        self.assertTrue(result["review_required"])


if __name__ == "__main__":
    unittest.main()
