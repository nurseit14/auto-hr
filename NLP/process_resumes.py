import csv
import hashlib
import json
from pathlib import Path

from NLP.cv_parser import parse_cv


ROOT = Path(__file__).resolve().parents[1]

RESUMES_DIR = Path.home() / "Desktop" / "resumes"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "resumes.csv"

SCHEMA_VERSION = "1.0"
TEXT_VERSION = "v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        (text or "").encode("utf-8")
    ).hexdigest()


def make_document_id(pdf_path: Path) -> tuple[str, str]:
    """
    Stable ID based on the original PDF bytes.

    Same PDF -> same document_id,
    regardless of its numeric resume_id.
    """
    source_hash = sha256_bytes(
        pdf_path.read_bytes()
    )

    document_id = f"doc_{source_hash[:16]}"

    return document_id, source_hash


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        RESUMES_DIR.glob("*.pdf"),
        key=lambda p: p.name.lower(),
    )

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in: {RESUMES_DIR}"
        )

    rows = []

    print("=" * 80)
    print("AUTO-HR FINAL RESUME PROCESSING")
    print("=" * 80)
    print(f"Found {len(pdf_files)} PDFs")
    print()

    for number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        resume_id = f"resume_{number:04d}"

        print(
            f"[{number:02d}/{len(pdf_files)}] "
            f"{resume_id} | {pdf_path.name}"
        )

        try:
            result = parse_cv(pdf_path)

            document_id, source_hash = make_document_id(
                pdf_path
            )

            final_text = result["text"]

            text_hash = sha256_text(
                final_text
            )

            layouts = result["layouts"]

            layout_string = ", ".join(
                f"p{x['page']}={x['layout']}"
                for x in layouts
            )

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "resume_id": resume_id,
                    "document_id": document_id,
                    "filename": pdf_path.name,
                    "source_sha256": source_hash,
                    "text_version": TEXT_VERSION,
                    "text_sha256": text_hash,
                    "layouts": layout_string,
                    "review_required": result[
                        "review_required"
                    ],
                    "raw_text": result[
                        "raw_text"
                    ],
                    "clean_text": result[
                        "clean_text"
                    ],
                    "text": final_text,
                    "skills": json.dumps(
                        result["skills"],
                        ensure_ascii=False,
                    ),
                    "skill_count": result[
                        "skill_count"
                    ],
                    "raw_length": len(
                        result["raw_text"]
                    ),
                    "clean_length": len(
                        result["clean_text"]
                    ),
                    "text_length": len(
                        final_text
                    ),
                }
            )

        except Exception as error:
            print(
                f"    ERROR: {error}"
            )

            raise

    fieldnames = [
        "schema_version",
        "resume_id",
        "document_id",
        "filename",
        "source_sha256",
        "text_version",
        "text_sha256",
        "layouts",
        "review_required",
        "raw_text",
        "clean_text",
        "text",
        "skills",
        "skill_count",
        "raw_length",
        "clean_length",
        "text_length",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Saved {len(rows)} resumes")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Schema version: {SCHEMA_VERSION}")
    print(f"Text version: {TEXT_VERSION}")


if __name__ == "__main__":
    main()
