from pathlib import Path

import pandas as pd

from NLP.cv_parser import parse_cv


ROOT_DIR = Path(__file__).resolve().parent.parent

# Папка с PDF-резюме на рабочем столе
RESUMES_DIR = (
    Path.home()
    / "Desktop"
    / "resumes"
)

# Куда сохранить общий CSV
OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "resumes.csv"
)


def main():

    print("=" * 70)
    print("AUTO-HR RESUME DATASET PROCESSOR")
    print("=" * 70)

    # Создаём папку для результата, если её ещё нет
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Ищем все PDF-файлы в Desktop/resumes
    pdf_files = sorted(
        RESUMES_DIR.glob("*.pdf")
    )

    print(
        f"\nResume folder:\n{RESUMES_DIR}"
    )

    print(
        f"\nFound {len(pdf_files)} PDF resumes."
    )

    if not pdf_files:
        print(
            "\nNo PDF files found in:"
        )
        print(RESUMES_DIR)
        return

    results = []
    failed = []

    for number, file_path in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"\n[{number}/{len(pdf_files)}] "
            f"Processing: {file_path.name}"
        )

        try:

            result = parse_cv(
                file_path
            )

            # Не используем имя человека как ID
            resume_id = (
                f"resume_{number:04d}"
            )

            results.append(
                {
                    "resume_id":
                        resume_id,

                    "filename":
                        file_path.name,

                    "raw_text":
                        result["raw_text"],

                    "clean_text":
                        result["clean_text"],

                    "text":
                        result["text"],

                    "skills":
                        ", ".join(
                            result["skills"]
                        ),

                    "skill_count":
                        result["skill_count"],

                    "raw_character_count":
                        len(result["raw_text"]),

                    "clean_character_count":
                        len(result["clean_text"]),

                    "character_count":
                        len(result["text"]),
                }
            )

            print(
                f"  Raw characters: "
                f"{len(result['raw_text'])}"
            )

            print(
                f"  Clean characters: "
                f"{len(result['clean_text'])}"
            )

            print(
                f"  Anonymized characters: "
                f"{len(result['text'])}"
            )

            print(
                f"  Skills detected: "
                f"{result['skill_count']}"
            )

        except Exception as error:

            print(
                f"  FAILED: {error}"
            )

            failed.append(
                {
                    "filename":
                        file_path.name,

                    "error":
                        str(error),
                }
            )

    # ----------------------------------------
    # Сохраняем успешно обработанные резюме
    # ----------------------------------------

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------
    # Статистика
    # ----------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal PDF files: "
        f"{len(pdf_files)}"
    )

    print(
        f"Successfully processed: "
        f"{len(results)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    if results:

        zero_text = (
            df["character_count"] == 0
        ).sum()

        zero_skills = (
            df["skill_count"] == 0
        ).sum()

        print(
            f"CVs with no extracted text: "
            f"{zero_text}"
        )

        print(
            f"CVs with no detected skills: "
            f"{zero_skills}"
        )

        print(
            f"Average detected skills: "
            f"{df['skill_count'].mean():.2f}"
        )

        print(
            f"Maximum detected skills: "
            f"{df['skill_count'].max()}"
        )

        print(
            f"Minimum detected skills: "
            f"{df['skill_count'].min()}"
        )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ----------------------------------------
    # Ошибки
    # ----------------------------------------

    if failed:

        print(
            "\nFiles that failed:"
        )

        for item in failed:

            print(
                f"- {item['filename']}: "
                f"{item['error']}"
            )


if __name__ == "__main__":
    main()