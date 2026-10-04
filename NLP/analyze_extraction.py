import re
from collections import Counter
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "vacancies_with_skills.csv"
)


# --------------------------------------------------
# Stop words
# --------------------------------------------------

STOP_WORDS = {
    # Russian
    "и", "в", "во", "на", "с", "со", "по", "для", "из",
    "от", "до", "за", "к", "ко", "о", "об", "при", "или",
    "а", "но", "что", "как", "это", "мы", "вы", "наш",
    "ваш", "его", "ее", "их", "не", "да", "нет", "быть",
    "работа", "работы", "работать", "опыт", "опыта",
    "знание", "знания", "знаний", "умение", "умения",
    "требования", "обязанности", "условия",
    "лет", "года", "год", "будет", "является",
    "также", "которые", "который", "которая",
    "компания", "компании", "команде", "команда",
    "разработка", "разработки",

    # English
    "the", "a", "an", "and", "or", "of", "to", "in",
    "for", "with", "on", "at", "from", "by", "as",
    "is", "are", "be", "will", "we", "you", "our",
    "your", "this", "that", "have", "has", "experience",
    "work", "working", "knowledge", "skills", "skill",
    "requirements", "responsibilities", "company",
    "team", "development",

    # Kazakh common words
    "және", "мен", "үшін", "бұл", "бар", "жұмыс",
    "жұмысқа", "тәжірибе", "талаптар"
}


# --------------------------------------------------
# Text processing
# --------------------------------------------------

def tokenize(text):
    """
    Convert text into useful lowercase tokens.
    Supports Latin and Cyrillic text.
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    # Keep words such as c++, c#, .net, node.js
    tokens = re.findall(
        r"[a-zа-яёәіңғүұқөһ0-9][a-zа-яёәіңғүұқөһ0-9+#.\-/]*",
        text,
        flags=re.IGNORECASE,
    )

    cleaned_tokens = []

    for token in tokens:

        token = token.strip(".,;:!?()[]{}\"'")

        if not token:
            continue

        if token in STOP_WORDS:
            continue

        if len(token) < 2:
            continue

        # Ignore tokens that contain only numbers
        if token.isdigit():
            continue

        cleaned_tokens.append(token)

    return cleaned_tokens


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():

    print("=" * 70)
    print("NLP SKILL EXTRACTION ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} vacancies.")

    # Make sure skill_count exists
    if "skill_count" not in df.columns:
        raise ValueError(
            "Column 'skill_count' was not found. "
            "Run NLP.process_vacancies first."
        )

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    with_skills = df[df["skill_count"] > 0]
    without_skills = df[df["skill_count"] == 0]

    total = len(df)

    coverage = (
        len(with_skills) / total * 100
        if total > 0
        else 0
    )

    print("\n--- COVERAGE ---")

    print(f"Total vacancies: {total}")
    print(f"With detected skills: {len(with_skills)}")
    print(f"Without detected skills: {len(without_skills)}")
    print(f"Coverage: {coverage:.2f}%")

    # --------------------------------------------------
    # Most common detected skills
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP DETECTED SKILLS")
    print("=" * 70)

    all_skills = []

    for value in with_skills["skills"].dropna():

        skills = [
            skill.strip()
            for skill in str(value).split(",")
            if skill.strip()
        ]

        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    for skill, count in skill_counts.most_common(30):
        print(f"{skill:<25} {count}")

    # --------------------------------------------------
    # Analyze zero-skill vacancies
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COMMON WORDS IN VACANCIES WITH ZERO DETECTED SKILLS")
    print("=" * 70)

    word_counter = Counter()

    for text in without_skills["text"].fillna(""):
        word_counter.update(tokenize(text))

    for word, count in word_counter.most_common(60):
        print(f"{word:<30} {count}")

    # --------------------------------------------------
    # Titles of failed vacancies
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COMMON TITLES WITH ZERO DETECTED SKILLS")
    print("=" * 70)

    title_counts = Counter(
        without_skills["title"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    for title, count in title_counts.most_common(30):
        print(f"{count:>4}  {title}")

    # --------------------------------------------------
    # Show representative failed examples
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLES WITH ZERO DETECTED SKILLS")
    print("=" * 70)

    sample_size = min(20, len(without_skills))

    # Fixed random_state makes results reproducible
    examples = without_skills.sample(
        n=sample_size,
        random_state=42
    )

    for number, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        print(f"\n--- Example {number} ---")

        print(
            "Title:",
            row.get("title", "UNKNOWN")
        )

        print(
            "Job category:",
            row.get("Job", "UNKNOWN")
        )

        print(
            "City:",
            row.get("city", "UNKNOWN")
        )

        text = str(
            row.get("text", "")
        ).replace("\n", " ")

        # Keep terminal output readable
        if len(text) > 500:
            text = text[:500] + "..."

        print("Text:")
        print(text)

    # --------------------------------------------------
    # Save failed vacancies
    # --------------------------------------------------

    output_file = (
        ROOT_DIR
        / "data"
        / "processed"
        / "vacancies_without_skills.csv"
    )

    without_skills.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS FINISHED")
    print("=" * 70)

    print(
        f"\nZero-skill vacancies saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()
