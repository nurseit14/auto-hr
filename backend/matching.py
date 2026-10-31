"""Adapters for the existing NLP → ML1 signals → trained ML2 pipeline.

The offline route never impersonates multilingual semantic embeddings. Its
lexical shortlist and experimental model are reported separately in the API.
"""
from collections import Counter
import csv
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
import re

import numpy as np

from ML1.skill_similarity import calculate_skill_similarity
from ML2.features import extract_features, FEATURE_NAMES, professional_text
from ML2.ranker import LinearRanker
from NLP.skill_extractor import extract_skills
from NLP.skills_dictionary import SKILL_ALIASES

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_NOTE = "Учебный архив вакансий Казахстана за 2023 год. Доступность позиций и зарплат не проверяется в реальном времени."
WEAK_BEFORE = re.compile(r"(?:\b(?:learning|beginner(?: in)?|basic(?: knowledge of)?|no experience(?: with| in)?|never used)\b|изучаю|изучить|осваиваю|базов\w*(?: знани\w*)?|не работал(?: с| в)?|не использовал|нет опыта(?: работы)?(?: с| в)?|не знаю|знаком(?:ство)? с)\s*$", re.I)
WEAK_AFTER = re.compile(r"^\s*[:—–-]?\s*(?:только\s+)?(?:изучаю|осваиваю|не работал|не использовал|не знаю|нет опыта|\b(?:beginner|basic|learning|no experience|never used)\b)", re.I)


def uncertain_mention(text, skill):
    """A weak statement must govern this skill, never every word on its line.

    In particular, the professional term 'Machine Learning' is not the action
    'learning Docker'. Separate clauses stop one negation leaking to a neighbor.
    """
    occurrences = []
    for alias in [skill, *SKILL_ALIASES.get(skill, [])]:
        for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text, flags=re.I):
            before = text[:match.start()]
            after = text[match.end():]
            if re.search(r"(?:не знаю|не работал|не использовал|изучаю|осваиваю|never used|no experience)\s*,\s*$", before, re.I):
                before = before.rsplit(",", 1)[-1]
            # Shared list predicates ('PostgreSQL и Kubernetes не работал',
            # 'learning Python, Docker') apply to every member of that list.
            for other in set(extract_skills(before + " " + after)) - {skill}:
                for name in sorted([other, *SKILL_ALIASES.get(other, [])], key=len, reverse=True):
                    pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
                    before = re.sub(pattern, " ", before, flags=re.I)
                    after = re.sub(pattern, " ", after, flags=re.I)
            before = re.sub(r"(?:\s|[,/&]|\b(?:and|и)\b)+$", " ", before, flags=re.I)
            after = re.sub(r"^(?:\s|[,/&]|\b(?:and|и)\b)+", " ", after, flags=re.I)
            occurrences.append(bool(WEAK_BEFORE.search(before) or WEAK_AFTER.search(after)))
    return bool(occurrences) and all(occurrences)


def normalize_skill(value):
    found = extract_skills(value)
    return found[0] if len(found) == 1 else value.strip().casefold()


def public_candidate(record):
    return {key: record[key] for key in ("id", "name", "filename", "text", "skills", "notes", "is_demo", "created_at", "clarifications")}


def refresh_skills(record):
    auto = set(extract_skills(professional_text(record["text"])))
    for answer in record.get("clarifications", []):
        auto.update(extract_skills(professional_text(answer["answer"])))
    record["auto_skills"] = sorted(auto)
    record["skills"] = sorted((auto - set(record.get("manual_removed", []))) | set(record.get("manual_added", [])))
    return record


def skill_sources(record):
    sources = []
    for source, text in [("resume", record["text"]), *[("clarification", row["answer"]) for row in record["clarifications"]]]:
        # Use exactly the professional filtering used by the model. Keep the
        # original casing in quotes whenever a retained source line is available.
        originals = {professional_text(line).strip(): line for line in text.splitlines() if professional_text(line).strip()}
        for cleaned in professional_text(text).splitlines():
            original = originals.get(cleaned.strip(), cleaned)
            for line in re.split(r"(?<=[.!?;])\s+|,?\s+(?:но|однако|but|however)\s+", original, flags=re.I):
                for skill in extract_skills(professional_text(line)):
                    if skill in record["skills"]:
                        sources.append({"skill": skill, "quote": line.strip()[:400], "source": source, "uncertain": uncertain_mention(line, skill)})
    # A human correction is explicit evidence and is distinguishable from CV text.
    for skill in record.get("manual_added", []):
        sources.append({"skill": skill, "quote": f"HR добавил навык: {skill}", "source": "hr", "uncertain": False})
    # Re-confirming an extracted weak skill in the editor can make it explicit.
    for skill in record.get("manual_confirmed", []):
        sources.append({"skill": skill, "quote": f"HR подтвердил навык: {skill}", "source": "hr", "uncertain": False})
    return sources


def candidate_source(record):
    sources = skill_sources(record)
    strong = {row["skill"] for row in sources if not row["uncertain"]}
    weak = {row["skill"] for row in sources if row["uncertain"]} - strong
    excluded = set(record.get("manual_removed", [])) | weak
    text = professional_text("\n".join([record["text"], *[row["answer"] for row in record["clarifications"]]]))
    for skill in excluded:
        for alias in sorted([skill, *SKILL_ALIASES.get(skill, [])], key=len, reverse=True):
            text = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", "[навык не подтверждён]", text, flags=re.I)
    added = sorted(set(record.get("manual_added", [])) | set(record.get("manual_confirmed", [])))
    if added:
        text += "\nПодтвержденные HR навыки: " + ", ".join(added)
    return text, sources, strong, weak


class Matching:
    def __init__(self):
        self.catalogue = self._read_catalogue()
        self.by_id = {row["id"]: row for row in self.catalogue}
        self.model = None
        self.warning = ""
        try:
            self.model = LinearRanker.load(Path(__file__).parent / "models/ranker.json")
        except (OSError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).warning("ML2 unavailable: %s", exc)
            self.warning = "Модель ML2 недоступна; используется сравнение навыков и текста."
        self.ml1 = None
        if os.getenv("AUTO_HR_ENABLE_ML1") == "1":
            try:
                # Explicit opt-in only; sentence-transformers is an optional dependency.
                from ML1.matcher import JobMatcher
                self.ml1 = JobMatcher()
            except Exception as exc:
                logging.getLogger(__name__).warning("ML1 unavailable: %s", type(exc).__name__)
                self.warning = "ML1 не загружен; работает локальный подбор и ML2."
        self.doc_tokens = [Counter(self.tokens(self.vacancy_text(v))) for v in self.catalogue]
        self.df = Counter(token for doc in self.doc_tokens for token in doc)
        self.avg_length = sum(sum(doc.values()) for doc in self.doc_tokens) / max(1, len(self.doc_tokens))

    @staticmethod
    def tokens(text):
        return re.findall(r"[\w+#.]{2,}", professional_text(text), flags=re.UNICODE)

    @staticmethod
    def vacancy_text(vacancy):
        return "\n".join(str(vacancy.get(key, "")) for key in ("title", "requirements", "description", "responsibilities"))

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_catalogue():
        rows = []
        with (ROOT / "data/processed/vacancies_with_skills.csv").open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                description = row.get("text", "")
                full = "\n".join((row.get("title", ""), row.get("requirements", ""), description))
                rows.append({"id": str(row["vacancy_id"]), "title": row.get("title", ""),
                             "company": row.get("job", ""), "city": row.get("city", ""),
                             "salary": row.get("salary", ""), "experience": row.get("experience", ""),
                             "employment": row.get("employment", ""), "description": description,
                             "requirements": row.get("requirements", ""), "responsibilities": row.get("responsibilities", ""),
                             "skills": extract_skills(full), "url": row.get("url", ""),
                             "category": row.get("Job", "IT"), "is_custom": False, "publish_date": row.get("publish_date", "")})
        return rows

    def pipeline(self):
        engine = ("semantic-ml1" if self.ml1 else "lexical-skills") + ("+ml2" if self.model else "")
        return {"engine": engine,
                "label": ("NLP → ML1 → ML2" if self.ml1 else "NLP → навыки и текст → ML2") if self.model else "NLP → навыки и текст",
                "ml1_available": bool(self.ml1), "ml2_available": bool(self.model),
                "ml2_experimental": True, "score_semantics": "Относительный балл ранжирования, не вероятность трудоустройства",
                "warning": self.warning}

    def pair(self, record, vacancy):
        text, sources, strong, weak = candidate_source(record)
        need = set(vacancy["skills"])
        matched = sorted(need & strong)
        uncertain = sorted(need & weak)
        missing = sorted(need - strong - weak)
        # ML2 accepts original source keys; ensure description is included in text.
        features, facts = extract_features(text, {**vacancy, "text": vacancy["description"]})
        if self.model:
            score = float(self.model.predict(np.asarray([features]))[0])
        else:
            score = len(matched) / max(1, len(need))
        evidence = []
        for skill in matched + uncertain:
            candidates = [row for row in sources if row["skill"] == skill]
            chosen = sorted(candidates, key=lambda row: (row["uncertain"], row["source"] != "hr"))[0]
            evidence.append(chosen)
        if need:
            explanation = f"В тексте подтверждено {len(matched)} из {len(need)} обнаруженных навыков вакансии."
        else:
            explanation = "В вакансии не обнаружены технические навыки из словаря; проверьте требования вручную."
        if uncertain:
            explanation += " Требуют уточнения: " + ", ".join(uncertain) + "."
        if missing:
            explanation += " Не подтверждены в резюме: " + ", ".join(missing) + ". Это не означает отсутствие опыта."
        questions = [f"Какой у вас практический опыт с {skill}? Опишите задачу, свою роль и результат." for skill in (uncertain + missing)[:2]]
        if len(questions) < 2:
            questions.append(f"Какой проект лучше всего показывает ваш опыт для позиции «{vacancy['title']}»? Что вы сделали лично?")
        return {"candidate": public_candidate(record), "score": round(score, 6),
                "skill_score": len(matched) / len(need) if need else 0., "matched_skills": matched,
                "missing_skills": missing, "uncertain_skills": uncertain, "evidence": evidence,
                "explanation": explanation, "status": "new", "questions": questions,
                "signals": {**facts["signals"], "ml1_adjusted_skill_score": calculate_skill_similarity(matched, vacancy["skills"]),
                            "model_version": "ml2-pairwise-linear-v1" if self.model else "offline-baseline",
                            "model_features": FEATURE_NAMES}}

    def shortlist(self, records, vacancy, reviews):
        rows = [self.pair(record, vacancy) for record in records]
        statuses = {row["candidate_id"]: row["status"] for row in reviews}
        for row in rows:
            row["status"] = statuses.get(row["candidate"]["id"], "new")
        rows.sort(key=lambda row: (-row["score"], -row["skill_score"], row["candidate"]["id"]))
        return rows

    def matches(self, record, vacancies, limit):
        source, _, strong, _ = candidate_source(record)
        query = set(self.tokens(source))
        candidates = []
        semantic = {}
        pipeline = self.pipeline()
        if self.ml1:
            try:
                semantic = {str(row["vacancy_id"]): row["hybrid_score"] for row in self.ml1.match(source, top_k=len(self.catalogue), resume_skills=sorted(strong))}
            except Exception:
                logging.getLogger(__name__).exception("ML1 query failed; using local fallback")
                pipeline = {**pipeline, "engine": "lexical-skills" + ("+ml2" if self.model else ""),
                            "label": "NLP → навыки и текст → ML2" if self.model else "NLP → навыки и текст",
                            "ml1_available": False, "warning": "Семантический поиск ML1 не выполнился; для этого запроса использован локальный подбор."}
        for vacancy in vacancies:
            # Search ranks a bounded candidate pool, then the trained model reranks it.
            tokens = Counter(self.tokens(self.vacancy_text(vacancy)))
            length = sum(tokens.values())
            lexical = 0.
            for token in query & tokens.keys():
                df = self.df.get(token, 0)
                idf = math.log(1 + (len(self.catalogue) - df + .5) / (df + .5))
                tf = tokens[token]
                lexical += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / max(1, self.avg_length)))
            skill = len(strong & set(vacancy["skills"])) / max(1, len(vacancy["skills"]))
            baseline = lexical / max(1, len(query)) + 2 * skill
            if semantic:
                # Public source rows outside the actual semantic shortlist do
                # not compete on a different scale. Custom rows have no index,
                # so compare them on an explicitly bounded baseline score.
                if not vacancy["is_custom"] and vacancy["id"] not in semantic:
                    continue
                prefilter = semantic.get(vacancy["id"], min(1., baseline / 3))
            else:
                prefilter = baseline
            candidates.append((prefilter, vacancy))
        candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
        rows = []
        for _, vacancy in candidates[:max(60, limit * 3)]:
            assessment = self.pair(record, vacancy)
            rows.append({"vacancy": vacancy, **{key: assessment[key] for key in ("score", "skill_score", "matched_skills", "missing_skills", "uncertain_skills", "explanation")}})
        rows.sort(key=lambda row: (-row["score"], -row["skill_score"], row["vacancy"]["id"]))
        return rows[:limit], pipeline
