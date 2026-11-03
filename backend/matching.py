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
from ML2.features import extract_features, FEATURE_NAMES, FEATURE_VERSION, professional_text, uncertain_skill_mention
from ML2.ranker import LinearRanker, MODEL_VERSION
from NLP.skill_extractor import extract_skills
from NLP.skills_dictionary import SKILL_ALIASES

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_NOTE = "Учебный архив вакансий Казахстана за 2023 год. Доступность позиций и зарплат не проверяется в реальном времени."

FEATURE_LABELS = {
    "skill_coverage": "Навыки вакансии",
    "skill_jaccard": "Совпадение наборов навыков",
    "task_coverage": "Задачи вакансии",
    "role_compatibility": "Специализация",
    "seniority_compatibility": "Уровень в нужной специализации",
    "experience_compatibility": "Релевантный стаж",
    "education_level_compatibility": "Уровень образования",
    "education_field_coverage": "Направление образования",
    "language_compatibility": "Языки",
    "mandatory_skill_confirmation": "Обязательные навыки",
    "industry_coverage": "Отраслевой опыт",
    "communication_confirmation": "Коммуникация",
    "required_skill_coverage": "Нужные навыки без пожеланий",
    "demonstrated_skill_coverage": "Навыки в описанных задачах",
    "relevant_experience_fulfillment": "Выполнение профильного стажа",
}


def uncertain_mention(text, skill):
    """Use the same uncertainty policy for visible coverage and ML2 features."""
    return uncertain_skill_mention(text, skill)


def normalize_skill(value):
    found = extract_skills(value)
    return found[0] if len(found) == 1 else value.strip().casefold()


def public_candidate(record):
    return {**{key: record[key] for key in ("id", "name", "filename", "text", "skills", "notes", "is_demo", "created_at", "clarifications")},
            "vacancy_ids": record.get("vacancy_ids", [])}


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


def source_evidence(items, record, vacancy):
    """Recover original quotes and provenance from the model's joined input.

    The extractor receives a filtered concatenation of resume and answers. Its
    source hints therefore cannot establish that a quote came from the resume.
    Never expose a transformed/masked model string as an original source quote.
    """
    original_sources = [("resume", record["text"])]
    original_sources += [("clarification", item["answer"]) for item in record.get("clarifications", [])]
    added = sorted(set(record.get("manual_added", [])) | set(record.get("manual_confirmed", [])))
    if added:
        original_sources.append(("hr", "Подтвержденные HR навыки: " + ", ".join(added)))
    vacancy_sources = [("vacancy", str(vacancy.get(field, "")))
                       for field in ("title", "requirements", "responsibilities", "description", "experience")]
    output = []
    seen = set()
    for item in items or []:
        quote = item.get("quote", "") if isinstance(item, dict) else ""
        if not isinstance(quote, str) or not quote.strip():
            continue
        pattern = r"\s+".join(re.escape(part) for part in quote.strip().split())
        candidates = vacancy_sources if item.get("source") == "vacancy" else original_sources
        for source, text in candidates:
            match = re.search(pattern, text, re.I)
            if match:
                # Limit display length while retaining a real contiguous quote.
                actual = match.group()[:600]
                identity = (source, actual)
                if identity not in seen:
                    output.append({"quote": actual, "source": source})
                    seen.add(identity)
                break
    return output


def score_breakdown(model, features, facts, record, vacancy, matched, evidence):
    """Explain exactly the score used for ordering, without counterfactual claims."""
    feature_reasons = facts.get("feature_reasons", {})
    feature_evidence = facts.get("feature_evidence", {})
    if model is not None:
        # Use the model's own transform so explanation cannot drift from predict.
        if hasattr(model, "explain"):
            calculation = model.explain(np.asarray([features]))[0]
        else:
            # Compatible with saved pilot models during a source-only upgrade.
            values = np.asarray(features, dtype=float)
            parts = ((values - model.mean) / model.scale) * model.weights
            calculation = {"base_score": 0., "raw_score": float(parts.sum()),
                           "contributions": [{"feature": name, "value": float(values[i]),
                                              "mean": float(model.mean[i]), "scale": float(model.scale[i]),
                                              "weight": float(model.weights[i]), "contribution": float(parts[i])}
                                             for i, name in enumerate(FEATURE_NAMES)]}
        contributions = []
        for item in calculation["contributions"]:
            name = item["feature"]
            reason = feature_reasons.get(name)
            if not reason:
                reason = ("По этому признаку пока нет достаточного подтверждения в источниках."
                          if item["value"] == 0 else "Признак рассчитан по профессиональному тексту и требованиям вакансии.")
            if item["weight"] == 0:
                reason += " В текущей обученной модели этот признак имеет нулевой вес."
            contributions.append({**item, "label": FEATURE_LABELS.get(name, name), "reason": reason,
                                  "evidence": source_evidence(feature_evidence.get(name, []), record, vacancy)})
        version, feature_version = MODEL_VERSION, FEATURE_VERSION
        warnings = list(facts.get("warnings", []))
    else:
        score = len(matched) / max(1, len(vacancy["skills"]))
        calculation = {"base_score": 0., "raw_score": score}
        contributions = [{"feature": "skill_coverage", "label": FEATURE_LABELS["skill_coverage"],
                          "value": score, "mean": 0., "scale": 1., "weight": 1.,
                          "contribution": score,
                          "reason": f"Подтверждено {len(matched)} из {len(vacancy['skills'])} обнаруженных навыков вакансии.",
                          "evidence": [{"quote": item["quote"], "source": item["source"]}
                                       for item in evidence if item["skill"] in matched]}]
        version, feature_version = "offline-baseline", "dictionary-skills"
        warnings = ["ML2 недоступна. Балл рассчитан только по доле подтверждённых навыков."]
    adjustments = []
    for item in calculation.get("adjustments", []):
        relevant = facts.get("relevant_years")
        required = facts.get("minimum_general_years")
        reason = "Явное правило учитывает подтверждённый недостаток профильного стажа; это не обученный коэффициент."
        if relevant is not None and required is not None:
            years = f"{relevant:g}".replace(".", ",")
            minimum = f"{required:g}".replace(".", ",")
            factor = f"{item['factor']:.6f}".rstrip("0").rstrip(".").replace(".", ",")
            reason = (f"Подтверждённый профильный стаж: {years} г.; минимум вакансии: {minimum} г. "
                      f"Положительная часть обученного балла умножена на {factor}. "
                      "Это явное правило, а не обученный коэффициент; неизвестный стаж не считается нулевым.")
        proofs = list(facts.get("experience_evidence", []))
        if vacancy.get("experience"):
            proofs.append({"quote": vacancy["experience"], "source": "vacancy"})
        adjustments.append({**item, "label": "Недостаточный профильный стаж", "reason": reason,
                            "evidence": source_evidence(proofs, record, vacancy)})
    return {"model_version": version, "feature_version": feature_version,
            "base_score": float(calculation.get("base_score", 0.)),
            "raw_score": float(calculation["raw_score"]),
            "final_score": float(calculation.get("final_score", calculation["raw_score"])),
            "contributions": contributions, "adjustments": adjustments,
            "experience": {"general_years": facts.get("claimed_total_years"),
                           "relevant_years": facts.get("relevant_years"),
                           "required_years": facts.get("minimum_general_years"),
                           "reason": facts.get("experience_reason", "Релевантный стаж по этой вакансии не подтверждён."),
                           "evidence": source_evidence(facts.get("experience_evidence", []), record, vacancy)},
            "role": {"candidate_areas": facts.get("cv_professional_areas", []),
                     "vacancy_areas": facts.get("vacancy_professional_areas", []),
                     "reason": facts.get("role_reason", "Специализацию необходимо проверить по резюме.")},
            "warnings": warnings}


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
                "model_version": MODEL_VERSION if self.model else "offline-baseline",
                "feature_version": FEATURE_VERSION if self.model else "dictionary-skills",
                "warning": self.warning}

    def pair(self, record, vacancy):
        text, sources, strong, weak = candidate_source(record)
        need = set(vacancy["skills"])
        matched = sorted(need & strong)
        uncertain = sorted(need & weak)
        missing = sorted(need - strong - weak)
        # ML2 accepts original source keys; ensure description is included in text.
        features, facts = extract_features(text, {**vacancy, "text": vacancy["description"]})
        evidence = []
        for skill in matched + uncertain:
            candidates = [row for row in sources if row["skill"] == skill]
            chosen = sorted(candidates, key=lambda row: (row["uncertain"], row["source"] != "hr"))[0]
            evidence.append(chosen)
        breakdown = score_breakdown(self.model, features, facts, record, vacancy, matched, evidence)
        score = breakdown["final_score"]
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
                "score_breakdown": breakdown,
                "signals": {**facts["signals"], "ml1_adjusted_skill_score": calculate_skill_similarity(matched, vacancy["skills"]),
                            "model_version": breakdown["model_version"],
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
            rows.append({"vacancy": vacancy, **{key: assessment[key] for key in ("score", "skill_score", "matched_skills", "missing_skills", "uncertain_skills", "explanation", "score_breakdown")}})
        rows.sort(key=lambda row: (-row["score"], -row["skill_score"], row["vacancy"]["id"]))
        return rows[:limit], pipeline
