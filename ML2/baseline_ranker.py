"""ML2-04 fixed, source-grounded reranking of an existing ML1 shortlist.

These engineering weights are declared, not fitted or tuned. The baseline
does not load the trained ML2 model, annotations, external services or a corpus.
Unknown evidence earns no contribution and is never labelled as inability.
"""

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType

from ML2.features import (
    FEATURE_NAMES, FEATURE_VERSION, OPTIONAL_REQUIREMENT, SKILL_ALIASES, extract_features,
    feature_fingerprint, language_levels, professional_text, required_skill_evidence,
    required_years, roles, skill_evidence, source_vacancy, statements,
    uncertain_skill_mention,
)
from NLP.skill_extractor import extract_skills


BASELINE_VERSION = "ml2-transparent-baseline-v1"
BASELINE_WEIGHTS = MappingProxyType({
    "ml1_hybrid": .25,
    "required_skills": .30,
    "demonstrated_skills": .15,
    "professional_profile": .15,
    "relevant_experience": .10,
    "languages": .05,
})
WEIGHTS = BASELINE_WEIGHTS
WEIGHTS_SOURCE = "Fixed engineering weights; not fitted or tuned on labels"
SCORE_SEMANTICS = "Relative evidence score, not a probability or annotation grade"
EFFECTIVE_SOURCE_POLICY = (
    "Source texts stay unchanged. For this baseline only, a statement containing "
    "a required skill with contradictory positive and explicit negative claims "
    "or explicit non-use missed by the shared feature extractor is excluded from "
    "professional feature extraction. This conservative policy "
    "can also exclude other facts in that statement; exclusions are reported."
)
_OUTPUT_KEYS = frozenset({"baseline_score", "baseline_rank", "ml1_rank",
                          "baseline_version", "baseline_breakdown", "baseline_evidence"})
_FEATURE_KEYS = {
    "required_skills": "required_skill_coverage",
    "demonstrated_skills": "demonstrated_skill_coverage",
    "professional_profile": "role_compatibility",
    "relevant_experience": "experience_compatibility",
    "languages": "language_compatibility",
}
_LABELS = {
    "ml1_hybrid": "Исходное сходство ML1",
    "required_skills": "Нужные навыки без пожеланий",
    "demonstrated_skills": "Навыки в описанных задачах",
    "professional_profile": "Профессиональный профиль",
    "relevant_experience": "Профильный стаж",
    "languages": "Языковые уровни",
}
_NEGATIVE_BEFORE = re.compile(
    r"(?:не\s+(?:использовал\w*|использую|работал\w*|работаю|знаю|применял\w*|применяю)|"
    r"нет\s+опыта(?:\s+работы)?|\bno\s+(?:work\s+)?experience|"
    r"\bnever\s+(?:used|worked|applied)|"
    r"\b(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't)"
    r"\s+(?:use[ds]?|work(?:ed)?|apply|applied))"
    r"\s*(?:(?:с|в|with|in|of)\s*)?(?:[,/&]|\b(?:and|or|и|или)\b|\s)*$", re.I,
)
_NEGATIVE_AFTER = re.compile(
    r"^\s*[:—–-]?\s*(?:не\s+(?:использовал\w*|использую|работал\w*|работаю|знаю|применял\w*|применяю)|"
    r"нет\s+опыта|(?:опыт\w*|практик\w*)[^.!?;]{0,50}\bнет\b|"
    r"\bno\s+experience|\bnever\s+(?:used|worked|applied)|"
    r"(?:(?:is|was)\s+(?:a[n]?\s+)?(?:tool|technology|framework|language|database)"
    r"\s+(?:(?:that|which)\s+)?)?(?:(?:i|we|he|she)\s+)?"
    r"(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't|never)"
    r"\s+(?:use[ds]?|work(?:ed)?|apply|applied))", re.I,
)
_LANGUAGE_ASPIRATION = re.compile(
    r"\b(?:want|hope|plan|aim)\s+to\b|\btarget\b|\bgoal\b|\baspir\w*|"
    r"\b(?:learning|studying)\b|хочу|планирую|намерен|изучаю|осваиваю|"
    r"\bцель\b|целев\w*|желаем\w*", re.I,
)
_LANGUAGE_DENIAL = re.compile(
    r"не\s+(?:владею|знаю|говорю|достиг\w*)|"
    r"\b(?:cannot|can't|don't|do not)\s+(?:speak|know|use)\b|"
    r"\b(?:no|without)\s+(?:knowledge|proficiency)\b", re.I,
)


def baseline_config():
    """Return a new serializable description of the fixed formula."""
    return {"version": BASELINE_VERSION, "weights": dict(BASELINE_WEIGHTS),
            "weights_source": WEIGHTS_SOURCE, "score_semantics": SCORE_SEMANTICS,
            "unknown_policy": "neutral zero; no weight renormalization",
            "tie_policy": "preserve input ML1 order",
            "conflict_policy": EFFECTIVE_SOURCE_POLICY}


def _aliases(skill):
    return sorted({skill, *SKILL_ALIASES.get(skill, [])}, key=len, reverse=True)


def _explicit_denial(part, skill):
    """Recognize direct non-use, not learning or an absent word.

    The shared uncertainty predicate scopes indirect denied technology lists.
    Other technology names are removed only from the local grammatical context
    so a denied list can be recognised without transferring a neighbour's claim.
    """
    for alias in _aliases(skill):
        for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", part, re.I):
            before, after = part[:match.start()], part[match.end():]
            if _NEGATIVE_AFTER.search(after):
                return True
            if (re.search(r"опыт\w*|практик\w*", before, re.I)
                    and re.match(r"^\s*[:—–-]?\s*нет\b", after, re.I)):
                return True
            for other in set(extract_skills(before)) - {skill}:
                for name in _aliases(other):
                    before = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", before, flags=re.I)
            if _NEGATIVE_BEFORE.search(before):
                return True
            # Handle a directly denied technology list without mistaking
            # "used Python, but not Docker" for a denial of Python.
            if uncertain_skill_mention(part, skill):
                for other in set(extract_skills(after)) - {skill}:
                    for name in _aliases(other):
                        after = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", after, flags=re.I)
                after = re.sub(r"^(?:\s|[,/&]|\b(?:and|or|и|или)\b)+", " ", after, flags=re.I)
                if _NEGATIVE_AFTER.search(after):
                    return True
    return False


def _original_quote(quote, source):
    """Restore source casing; never present a generated string as a quote."""
    if not quote:
        return None
    if quote in source:
        return quote
    folded, positions = [], []
    for index, char in enumerate(source):
        expanded = char.casefold()
        folded.append(expanded)
        positions.extend([index] * len(expanded))
    needle = quote.casefold()
    start = "".join(folded).find(needle)
    if start >= 0:
        return source[positions[start]:positions[start + len(needle) - 1] + 1]
    # Contact masking may have removed a substring inside a supporting span.
    # Keep the original continuous statement instead of fabricating its quote.
    for part in statements(source):
        if needle in professional_text(part):
            return part
    return None


def _proofs(items, source, source_name="resume"):
    output = []
    for item in items:
        quote = _original_quote(item.get("quote", ""), source)
        if quote and not any(row["quote"] == quote for row in output):
            output.append({"source": source_name, "quote": quote})
    return output


def _skill_assessments(cv_text, required):
    positive = skill_evidence(professional_text(cv_text))
    raw_parts = statements(cv_text)
    output, conflicts = [], []
    for skill, vacancy_proofs in sorted(required.items()):
        confirmed = _proofs(positive.get(skill, []), cv_text)
        confirmed = [proof for proof in confirmed if not _explicit_denial(professional_text(proof["quote"]), skill)]
        denied = [{"source": "resume", "quote": part} for part in raw_parts
                  if skill in extract_skills(part) and _explicit_denial(professional_text(part), skill)]
        conflict = bool(confirmed and denied)
        status = "unknown" if conflict or not (confirmed or denied) else "match" if confirmed else "mismatch"
        reason = ("В резюме есть противоречащие друг другу утверждения; опыт требует уточнения."
                  if conflict else "Навык прямо указан в профессиональной части резюме."
                  if confirmed else "Есть прямое утверждение об отсутствии опыта применения."
                  if denied else "Достаточных подтверждений нет; отсутствие навыка не установлено.")
        row = {"skill": skill, "status": status, "conflict": conflict,
               "reason": reason, "evidence": confirmed + denied,
               "vacancy_evidence": deepcopy(vacancy_proofs)}
        output.append(row)
        if conflict:
            conflicts.append({"skill": skill, "positive_evidence": confirmed,
                              "negative_evidence": denied})
    return output, conflicts


def _required_language_levels(fields):
    """Apply the existing explicit-wish policy before parsing language levels."""
    kept, optional_section = [], False
    for part in statements(professional_text(fields["requirements"])):
        if re.match(r"(?:обязател\w*|требован\w*|must[ -]have|required)\s*[:：]", part, re.I):
            optional_section = False
        optional = OPTIONAL_REQUIREMENT.search(part)
        if optional:
            prefix = part[:optional.start()].rstrip(" ,;:—–-")
            trailing = part[optional.end():].strip(" ().,;:—–-")
            if prefix and not trailing:
                prior = prefix.rsplit(",", 1)[0] if "," in prefix else ""
                prefix = prior if re.search(r"\brequired\b|\bmust\b|обязател\w*", prior, re.I) else ""
            if prefix and not optional_section:
                kept.append(prefix)
            optional_section = not prefix and bool(re.search(r"[:：]\s*$", part))
            continue
        if not optional_section:
            kept.append(part)
    return language_levels("\n".join(kept))


def _language_assessments(effective, fields, context):
    required = _required_language_levels(fields)
    affirmed, denied, uncertain = {}, {}, {}
    for part in statements(effective):
        levels = language_levels(professional_text(part))
        target = (uncertain if _LANGUAGE_ASPIRATION.search(part) else
                  denied if _LANGUAGE_DENIAL.search(part) else affirmed)
        for name, level in levels.items():
            target.setdefault(name, []).append({"level": level,
                                                "evidence": {"source": "resume", "quote": part}})
    rows, scores = [], []
    for name, needed in sorted(required.items()):
        positive = affirmed.get(name, [])
        negative = denied.get(name, [])
        weak = uncertain.get(name, [])
        candidates = {row["level"] for row in positive}
        conflict = bool((positive and negative) or len(candidates) > 1)
        actual = next(iter(candidates)) if len(candidates) == 1 and not conflict else None
        if conflict:
            status, reason, value = "unknown", "Языковые утверждения противоречат друг другу; уровень требует уточнения.", 0.
        elif negative:
            status, reason, value = "mismatch", "Есть прямое отрицание владения указанным уровнем; фактический уровень не выдумывается.", 0.
        elif actual is None:
            status, reason, value = "unknown", "Подтверждённый текущий уровень не указан; цель обучения не является уровнем владения.", 0.
        else:
            status = "match" if actual >= needed else "mismatch"
            reason = "Сравниваются явно указанный текущий уровень и уровень вакансии."
            value = 1. if actual >= needed else -(needed - actual) / 6
        scores.append(value)
        rows.append({"language": name, "status": status, "candidate_level": actual,
                     "required_level": needed, "conflict": conflict, "reason": reason,
                     "evidence": [row["evidence"] for row in positive + negative + weak]})
    # Unknown requested languages contribute zero, rather than letting the one
    # available language earn the whole language component.
    value = math.fsum(scores) / len(required) * context if required else 0.
    reason = ("Сопоставляются текущие явно указанные уровни. Неизвестные запрошенные языки дают ноль; знаменатель включает все запрошенные уровни."
              if required else "Сопоставимые явно заданные уровни языков в требованиях не найдены; бонус не добавляется.")
    return value, {"status": _status_summary(rows), "requirements": rows, "reason": reason}


def _minimum_years(fields):
    text = fields["experience"]
    if not text.strip():
        return None, "not_specified", "Минимальный стаж не указан в структурированном поле опыта."
    if re.search(r"нет опыта|без опыта|no experience", text, re.I):
        return 0., "not_required", "Вакансия явно не требует стажа; бонус за стаж не добавляется."
    if (re.search(r"\bдо\s+\d|\bup\s+to\b|\bat\s+most\b|не\s+более", text, re.I)
            and not re.search(r"\bот\s+\d|\bfrom\s+\d|\bat\s+least\b|не\s+менее", text, re.I)):
        return None, "unknown", "Указана только верхняя граница стажа; она не трактуется как минимум."
    if re.search(r"months?|месяц", text, re.I) or not re.search(r"\byears?\b|\bлет\b|\bгод\w*", text, re.I):
        return None, "unknown", "Единицы или формулировка минимума стажа не поддержаны; число не трактуется как годы."
    value = required_years(text)
    if value is None:
        return None, "unknown", "Минимум профильных лет не удалось однозначно извлечь."
    return value, None, ""


def _status_summary(rows):
    statuses = {row["status"] for row in rows}
    if not statuses:
        return "not_specified"
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"


def _validate_score(row, name, required=False):
    if name not in row:
        if required:
            raise ValueError("ML1 hybrid_score is required; no replacement score is invented")
        return
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError(f"ML1 {name} must be a finite number in [0, 1]")


class TransparentBaseline:
    """A deterministic baseline with the same declared formula for every pair."""

    @staticmethod
    def config():
        return baseline_config()

    def score_pair(self, cv_text, vacancy):
        """Explain one ML1 candidate without mutating its fields or source text."""
        if not isinstance(cv_text, str) or not professional_text(cv_text).strip():
            raise ValueError("A nonempty professional source CV is required")
        if not isinstance(vacancy, dict):
            raise ValueError("Each ML1 result must be a dictionary")
        for name in ("hybrid_score", "skill_score", "semantic_score"):
            _validate_score(vacancy, name, required=name == "hybrid_score")
        if vacancy.get("vacancy_fields") is not None and not isinstance(vacancy["vacancy_fields"], dict):
            raise ValueError("vacancy_fields must be a source dictionary")
        source = vacancy.get("vacancy_fields") or vacancy
        for name in ("title", "experience", "requirements", "responsibilities", "text", "description"):
            if source.get(name) is not None and not isinstance(source[name], str):
                raise ValueError(f"Source field {name} must be text or null")
        if vacancy.get("vacancy_title") is not None and not isinstance(vacancy["vacancy_title"], str):
            raise ValueError("Source vacancy_title must be text or null")
        fields = source_vacancy(vacancy)
        vacancy_source = "\n".join(fields.values())
        required = required_skill_evidence(fields)
        requirements, conflicts = _skill_assessments(cv_text, required)
        for row in requirements:
            row["vacancy_evidence"] = _proofs(row["vacancy_evidence"], vacancy_source, "vacancy")
        conflicted = {row["skill"] for row in conflicts}
        excluded = []
        for part in statements(cv_text):
            mentioned = set(extract_skills(part)) & set(required)
            legacy_positive = skill_evidence(professional_text(part))
            missed_denial = any(skill in legacy_positive and _explicit_denial(professional_text(part), skill)
                                for skill in mentioned)
            if mentioned & conflicted or missed_denial:
                excluded.append(part)
        effective = "\n".join(part for part in statements(cv_text) if part not in excluded)
        if effective.strip() and professional_text(effective).strip():
            values, facts = extract_features(effective, vacancy)
            signals = dict(zip(FEATURE_NAMES, values))
        else:
            signals = dict.fromkeys(FEATURE_NAMES, 0.)
            facts = {"relevant_years": None, "relevance_supported": False,
                     "feature_reasons": {}, "feature_evidence": {}, "warnings": [],
                     "cv_professional_areas": [], "vacancy_professional_areas": sorted(roles(fields["title"]))}
        minimum, minimum_status, minimum_reason = _minimum_years(fields)
        years = facts["relevant_years"]
        warnings = list(facts["warnings"])
        if excluded:
            warnings.append("Противоречивые утверждения о навыках и прямые отрицания, не распознанные общим извлекателем, исключены из признаков только этого baseline; другие факты в этих же высказываниях также могли быть исключены.")
        if minimum_status:
            experience_status, experience_reason = minimum_status, minimum_reason
            signals["experience_compatibility"] = 0.
            if minimum_status == "unknown":
                warnings.append(minimum_reason)
        elif years is None or not facts["relevance_supported"]:
            experience_status = "unknown"
            experience_reason = "Продолжительность профильного опыта не установлена; это не нулевой стаж."
            signals["experience_compatibility"] = 0.
        else:
            experience_status = "match" if years >= minimum else "mismatch"
            experience_reason = facts.get("feature_reasons", {}).get("experience_compatibility", "Сравниваются явно указанные профильные годы и минимум вакансии.")
        evidence_by_feature = {
            name: _proofs(items, cv_text) for name, items in facts.get("feature_evidence", {}).items()
        }
        experience = {"status": experience_status, "relevant_years": years,
                      "required_years": minimum, "reason": experience_reason,
                      "evidence": evidence_by_feature.get("experience_compatibility", [])}
        context = max(signals["role_compatibility"], float(facts.get("technical_support", 0.)))
        language_value, language_evidence = _language_assessments(effective, fields, context)
        signals["language_compatibility"] = language_value
        language_status, language_reason = language_evidence["status"], language_evidence["reason"]
        evidence_by_feature["language_compatibility"] = [proof for row in language_evidence["requirements"]
                                                         for proof in row["evidence"]]
        profile_status = "match" if facts["relevance_supported"] else "unknown"
        profile = {"status": profile_status,
                   "reason": facts.get("role_reason") or "Роль и связанные технические свидетельства не подтверждены одновременно.",
                   "evidence": evidence_by_feature.get("role_compatibility", []),
                   "candidate_professional_areas": facts["cv_professional_areas"],
                   "vacancy_professional_areas": facts["vacancy_professional_areas"]}
        skill_status = _status_summary(requirements)
        reasons = facts.get("feature_reasons", {})
        statuses = {"ml1_hybrid": "provided", "required_skills": skill_status,
                    "demonstrated_skills": "not_specified" if not required else "match" if signals["demonstrated_skill_coverage"] == 1 else "partial" if signals["demonstrated_skill_coverage"] > 0 else "unknown",
                    "professional_profile": profile_status,
                    "relevant_experience": experience_status, "languages": language_status}
        component_reasons = {
            "ml1_hybrid": "Используется предоставленный hybrid_score ML1; его формула и исходные баллы сохраняются без пересчёта.",
            "required_skills": reasons.get("required_skill_coverage", "Нет подтверждённых нужных навыков.") + " Это покрытие текста, а не доказательство отсутствия остальных навыков.",
            "demonstrated_skills": reasons.get("demonstrated_skill_coverage", "Применение нужных навыков в задачах не подтверждено."),
            "professional_profile": profile["reason"], "relevant_experience": experience_reason,
            "languages": language_reason,
        }
        components = []
        for key, weight in BASELINE_WEIGHTS.items():
            value = float(vacancy["hybrid_score"]) if key == "ml1_hybrid" else float(signals[_FEATURE_KEYS[key]])
            proof = ([{"source": "ml1", "field": "hybrid_score", "value": value}]
                     if key == "ml1_hybrid" else evidence_by_feature.get(_FEATURE_KEYS[key], []))
            components.append({"key": key, "label": _LABELS[key], "value": value,
                               "weight": weight, "contribution": weight * value,
                               "status": statuses[key], "reason": component_reasons[key],
                               "evidence": proof})
        total = math.fsum(row["contribution"] for row in components)
        config = baseline_config()
        config_fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        breakdown = {**config, "components": components, "total": total,
                     "feature_version": FEATURE_VERSION, "feature_fingerprint": feature_fingerprint(),
                     "baseline_source_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "config_fingerprint": config_fingerprint}
        evidence = {"requirements": requirements, "experience": experience,
                    "languages": language_evidence, "profile": profile,
                    "conflicts": conflicts, "effective_source_policy": EFFECTIVE_SOURCE_POLICY,
                    "excluded_statements": [{"source": "resume", "quote": part} for part in excluded],
                    "warnings": warnings}
        return {"baseline_score": total, "baseline_version": BASELINE_VERSION,
                "baseline_breakdown": breakdown, "baseline_evidence": evidence}

    def rerank(self, cv_text, vacancies, top_k=10):
        """Preserve every input field and retain ML1 input order on exact ties."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(cv_text, str) or not professional_text(cv_text).strip():
            raise ValueError("A nonempty professional source CV is required")
        if not isinstance(vacancies, list):
            raise ValueError("ML1 results must be a list")
        ids = set()
        for row in vacancies:
            if not isinstance(row, dict):
                raise ValueError("Each ML1 result must be a dictionary")
            collision = _OUTPUT_KEYS & row.keys()
            if collision:
                raise ValueError("Reserved baseline output fields already present: " + ", ".join(sorted(collision)))
            identity = row.get("vacancy_id") if "vacancy_id" in row else row.get("id")
            if isinstance(identity, bool) or not isinstance(identity, (str, int)) or not str(identity).strip():
                raise ValueError("Vacancy IDs must be present and unique")
            canonical = str(identity)
            if canonical in ids:
                raise ValueError("Vacancy IDs must be present and unique")
            ids.add(canonical)
        output = []
        for index, row in enumerate(vacancies, 1):
            result = deepcopy(row)
            if "vacancy_id" not in result:
                result["vacancy_id"] = str(row["id"])
            result.update(self.score_pair(cv_text, row))
            result["ml1_rank"] = index
            output.append(result)
        output.sort(key=lambda row: -row["baseline_score"])
        for rank, row in enumerate(output, 1):
            row["baseline_rank"] = rank
        return output[:top_k]
