"""Read existing AUTO-HR CSVs and select five distinct vacancies for every CV."""

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def digest(value) -> str:
    encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(stream)]


def skills(value: str) -> set[str]:
    if value.startswith("["):
        try:
            items = ast.literal_eval(value)
            if isinstance(items, list):
                return {str(x).strip().casefold() for x in items if str(x).strip()}
        except (ValueError, SyntaxError):
            pass
    return {x.strip().casefold() for x in re.split(r"[,;|]", value) if x.strip()}


def load_resumes(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("Resume CSV is empty")
    if "text" not in rows[0]:
        raise ValueError("Resume CSV needs a text column")
    output, seen = [], set()
    for row in rows:
        text = row["text"]
        cv_id = row.get("resume_id") or row.get("cv_id") or "cv_" + digest(text)[:16]
        if cv_id in seen:
            raise ValueError(f"Duplicate CV ID: {cv_id}")
        seen.add(cv_id)
        output.append({"cv_id": cv_id, "text": text, "skills": sorted(skills(row.get("skills", ""))),
                       "text_sha256": digest(text)})
    return output


def make_vacancy(row: dict) -> dict:
    fields = [
        ("Название", row.get("title", "")), ("Опыт", row.get("experience", "")),
        ("Требования", row.get("requirements", "")),
        ("Обязанности", row.get("responsibilities", "")),
    ]
    # Preserve the source text as well, including any facts absent in structured fields.
    text = row.get("text", "")
    if text:
        fields.append(("Сохранённый текст", text))
    display = "\n\n".join(f"{key}:\n{value}" for key, value in fields if value)
    if not display.strip():
        raise ValueError("Vacancy has no source text")
    vid = row.get("vacancy_id") or row.get("id") or "vac_" + digest(display)[:16]
    return {"vacancy_id": vid, "title": row.get("title", ""),
            "experience": row.get("experience", ""), "requirements": row.get("requirements", ""),
            "responsibilities": row.get("responsibilities", ""), "text": text,
            "display": display, "skills": sorted(skills(row.get("skills", ""))),
            "source_sha256": digest(display)}


def load_vacancies(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("Vacancy CSV is empty")
    output, seen = [], set()
    for row in rows:
        vacancy = make_vacancy(row)
        if vacancy["vacancy_id"] in seen:
            raise ValueError(f"Duplicate vacancy ID: {vacancy['vacancy_id']}")
        seen.add(vacancy["vacancy_id"])
        output.append(vacancy)
    return output


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w][\w+#.\-]*", text.casefold(), re.UNICODE)


def select_pairs(resumes: list[dict], vacancies: list[dict], count: int = 5) -> list[dict]:
    """Deterministic lexical/skill shortlist; this does not claim to be ML1."""
    if count < 1 or len(vacancies) < count:
        raise ValueError(f"Need at least {count} distinct vacancies")
    counts = [Counter(tokens(v["display"])) for v in vacancies]
    lengths = [sum(c.values()) for c in counts]
    mean_len = sum(lengths) / len(lengths)
    df = Counter(t for c in counts for t in c)
    idf = {t: math.log(1 + (len(vacancies) - n + .5) / (n + .5)) for t, n in df.items()}
    output = []
    for cv in resumes:
        query = set(tokens(cv["text"]))
        cv_skills = set(cv["skills"])
        scores = []
        for vacancy, freq, size in zip(vacancies, counts, lengths):
            bm25 = sum(idf[t] * freq[t] * 2.5 / (freq[t] + 1.5 * (.25 + .75 * size / mean_len))
                       for t in query.intersection(freq))
            vskills = set(vacancy["skills"])
            overlap = len(cv_skills & vskills) / max(1, len(vskills))
            scores.append((bm25 + 8 * overlap, vacancy))
        scores.sort(key=lambda item: (-item[0], item[1]["vacancy_id"]))
        # Avoid five copies of the same scraped publication where possible.
        chosen, sources = [], set()
        for score, vacancy in scores:
            fingerprint = vacancy["source_sha256"]
            if fingerprint not in sources:
                chosen.append((score, vacancy))
                sources.add(fingerprint)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise ValueError(f"Need {count} vacancies with different source content")
        for position, (score, vacancy) in enumerate(chosen, 1):
            identity = {"cv_id": cv["cv_id"], "vacancy_id": vacancy["vacancy_id"],
                        "cv_hash": cv["text_sha256"], "vacancy_hash": vacancy["source_sha256"]}
            output.append({"pair_id": "pair_" + digest(identity)[:20], "cv": cv, "vacancy": vacancy,
                           "selection_rank": position, "selection_score": round(score, 6),
                           "selection_method": "bm25-skills-v1"})
    return output
