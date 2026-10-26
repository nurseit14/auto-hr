"""Fixed professional signals extracted from source documents, never annotations.

This first feature model has no free-text vocabulary or personal-identity slots.
Unknown evidence remains unknown; missing technologies are not proof of inability.
"""

import hashlib
from pathlib import Path
import re

from NLP.skill_extractor import extract_skills


FEATURE_VERSION = "professional-signals-v1"
FEATURE_NAMES = ["skill_coverage", "skill_jaccard", "task_coverage", "role_compatibility",
                 "seniority_compatibility", "experience_compatibility",
                 "education_level_compatibility", "education_field_coverage",
                 "language_compatibility", "mandatory_skill_confirmation",
                 "industry_coverage", "communication_confirmation"]


def feature_fingerprint():
    root = Path(__file__).resolve().parent.parent
    source = b"".join((root / path).read_bytes() for path in
                      ("ML2/features.py", "NLP/skill_extractor.py", "NLP/skills_dictionary.py"))
    return hashlib.sha256(source).hexdigest()

ROLES = {
    "backend": r"backend|back.end|б[эе]к[эе]нд|(?:java|python|c#|\.net|golang|node\.?js)\s+(?:developer|разработчик)",
    "frontend": r"front.end|фронт.?енд|фронт.?энд",
    "fullstack": r"full[ -]?stack|фул[ -]?ст[еэ]к",
    "data": r"data\s+(?:scien|analyst|engineer)|machine learning|ml\s+(?:engineer|research)|power\s*bi|аналитик\s+данных",
    "business_analysis": r"business analyst|бизнес.?аналитик",
    "qa": r"\bqa\b|quality assurance|тестировщик|automation testing|инженер по тестированию",
    "devops": r"devops|site reliability|\bsre\b",
    "support": r"help\s*desk|it\s+(?:specialist|специалист)|it.специалист|техподдерж|системн\w*\s+администр|system\s+admin",
    "design": r"ui\s*[/&-]\s*ux|ux\s*[/&-]\s*ui|\bdesigner\b|дизайнер",
    "security": r"security|кибербезопас|информационн\w*\s+безопас",
    "sales": r"\bsales\b|продаж|business development|business relations",
    "marketing": r"marketing|маркетинг|монетизац|monetization",
    "finance": r"финанс\w*\s+аналитик|finance analyst|бухгалтер|accountant",
    "product": r"product\s+(?:manager|owner)|продукт\w*\s+менеджер|менеджер\s+продукт",
    "project": r"project\s+(?:manager|coordinator)|operations manager|руководител\w*\s+проект|менеджер\w*\s+проект",
    "teaching": r"преподавател|репетитор|\bteacher\b|\btutor\b|lecturer",
    "telecom": r"радио|телеком|\bррл\b|telecommunicat|network engineer",
    "software": r"software\s+(?:engineer|developer)|разработчик|программист|\bdeveloper\b",
}
TASKS = {
    "code": r"программир|разработк|programming|software development|write.*code|coding",
    "testing": r"тестирован|тест.план|testing|test.plans?|unit.tests?|\bqa\b",
    "design": r"дизайн|прототип|prototyp|wirefram|ui/ux|interface design",
    "data": r"анализ.*данн|data analys|\beda\b|визуализац|visualization|статистик|statistics",
    "documents": r"документац|documentation|technical writing|технич\w*\s+документ",
    "sales": r"продаж|\bsales\b|commercial proposal|коммерческ\w*\s+предлож",
    "clients": r"клиент|заказчик|client|customer|crm|партнер|партнёр|partner",
    "contracts": r"договор|contract",
    "leadership": r"руковод|управлен.*команд|lead.*team|team.*lead|supervis",
    "planning": r"планирован|project management|проект|budget|бюджет|roadmap|mvp",
    "tender": r"тендер|tender|procurement|закуп",
    "support": r"поддержк|support|troubleshoot|обслуживан",
    "training": r"обучен|препода|training|teaching|learning.plan",
    "marketing": r"маркетинг|marketing|реклам|advertis|promotion|продвижен",
    "deployment": r"deploy|ci/cd|production|разверт|развёрт",
    "networks": r"tcp/ip|маршрутиз|routing|subnet|ethernet|сетев|радио|telecom",
}
INDUSTRIES = {
    "finance": r"banking|finance|financial|accounting|банк|финанс|бухгалтер|\bopex\b",
    "education": r"education|образован|обучен|university|университет",
    "logistics": r"logistics|delivery|логистик|доставк|qazpost",
    "ecommerce": r"e.commerce|ecommerce|электронн\w*\s+коммерц|интернет.магазин",
    "mining": r"mining|горнодобыв|горноруд|горной\s+промышлен",
    "pharma": r"pharma|фармацевт|лекарств",
    "telecom": r"telecom|телеком|спутников|связи|связь",
}
EDUCATION_FIELDS = {
    "computing": r"computer science|software|information systems|информационн\w*\s+систем|программн\w*\s+обеспеч|\bit\b|\bит\b",
    "engineering": r"engineering|инженер|technical|техническ",
    "telecom": r"radio|радио|telecom|телеком|электроник|electronics",
    "business": r"business|бизнес|administration|enterprise management|менеджмент",
    "finance": r"finance|финанс|accounting|бухгалтер",
    "math": r"mathematics|математик|statistics|статистик",
}


def professional_text(text):
    """Remove direct contact/demographic lines and known template instructions."""
    lines = []
    for line in str(text).splitlines():
        if re.search(r"describe (?:what|the)|опишите (?:что|задачу)", line, re.I):
            if lines and re.search(r"project|проект", lines[-1], re.I):
                lines.pop()
            continue
        if re.search(r"гражданств|nationality|citizenship|date of birth|\bdob\b|родил|"
                     r"\b(?:женщина|мужчина)\b|\b(?:male|female)\b|marital|семейное положение|"
                     r"проживает|резюме обновлено|resume updated|история общения|preferred.*contact", line, re.I):
            continue
        line = re.sub(r"(?:https?://|www\.|t\.me/)\S+|[\w.+-]+@[\w.-]+\.\w+", " ", line)
        line = re.sub(r"(?:github|linkedin|phone|email|телефон|почта)\s*:\s*(?:\[[A-Z]+\]|\S+)?", " ", line, flags=re.I)
        line = re.sub(r"\[(?:PHONE|EMAIL|URL|CONTACT|NAME)\]", " ", line)
        lines.append(line)
    return "\n".join(lines).casefold()


def detected(text, patterns):
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def coverage(have, need):
    return len(have & need) / len(need) if need else 0.


def roles(text):
    found = detected(text, ROLES)
    # "developer" contributes a fallback family, not a second perfect match.
    if len(found) > 1:
        found.discard("software")
    return found


def seniority(text):
    found = []
    for level, pattern in [(0, r"\bintern\b|internship|стаж[её]р"),
                           (1, r"\bjunior\b|\bjun\b|младший|джун"),
                           (2, r"\bmiddle\b|\bmid.level\b|мидл"),
                           (3, r"\bsenior\b|\bsr\b|старший|сеньор"),
                           (4, r"\b(?:team|tech|technical)\s+lead\b|\blead\s+(?:developer|engineer|designer)\b|\bhead of\b|ведущий|руководитель")]:
        if re.search(pattern, text, re.I):
            found.append(level)
    return found


def claimed_years(text):
    # Only a claimed total; student dates and partially dated job histories do not
    # become guessed employment durations. No overlapping dates are summed.
    m = re.search(r"(?:опыт работы|work experience)\s*[:—–-]?\s*(\d+(?:[.,]\d+)?)\s*"
                  r"(?:лет|год\w*|years?)(?:\s+(\d+)\s*(?:месяц\w*|months?))?", text, re.I)
    if m:
        return float(m[1].replace(",", ".")) + float(m[2] or 0) / 12
    m = re.search(r"(?:with|over)\s+(\d+(?:\.\d+)?)\+?\s+years?\s+(?:of\s+)?experience", text, re.I)
    return float(m[1]) if m else None


def required_years(text):
    if re.search(r"нет опыта|без опыта|no experience", text, re.I):
        return 0.
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(m[0].replace(",", ".")) if m else None


def degree(text):
    levels = [(1, r"college|колледж|среднее специальное"),
              (2, r"\bbachelor\b|бакалавр|высшее образование|university degree|graduate degree"),
              (3, r"\bmaster\b|магистр|магистратур"), (4, r"\bph\.?d\b|doctorate|доктор наук")]
    found = [level for level, pattern in levels if re.search(pattern, text, re.I)]
    return max(found) if found else None


def language_levels(text):
    patterns = {"english": r"english|английск\w*|ағылшын\w*",
                "russian": r"russian|русск\w*|орыс\w*",
                "kazakh": r"kazakh|казахск\w*|қазақ\w*"}
    levels = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}
    output = {}
    for line in text.splitlines():
        matches = sorted((m.start(), m.end(), name) for name, pattern in patterns.items()
                         for m in re.finditer(pattern, line, re.I))
        prefix = line[:matches[0][0]] if matches else ""
        for i, (start, end, name) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(line)
            around = line[end:min(next_start, end + 80)]
            # A prefix such as "Fluent in English and Russian" applies to both.
            # A preceding language's C2 must never be borrowed by the next one.
            if not re.search(r"\b[abc][12]\b|native|родно|fluent|advanced|свобод|intermediate|good|хорош|еркін", around, re.I):
                around += " " + prefix
            explicit = re.search(r"\b([abc][12])\b", around, re.I)
            level = levels[explicit[1].casefold()] if explicit else None
            if level is None:
                for value, words in [(6, r"native|родно[йя]|ана тілі"),
                                     (5, r"fluent|advanced|свободн|в совершенстве|еркін"),
                                     (4, r"upper.intermediate|good|хорош"),
                                     (3, r"intermediate|средний|орташа")]:
                    if re.search(words, around, re.I):
                        level = value
                        break
            if level is not None:
                output[name] = level
    return output


def source_vacancy(row):
    """Adapt frozen source rows or ML1 result dictionaries without using labels."""
    fields = row.get("vacancy_fields") or row
    title = str(fields.get("title") or row.get("vacancy_title") or "")
    experience = str(fields.get("experience") or "")
    requirements = str(fields.get("requirements") or "")
    responsibilities = str(fields.get("responsibilities") or "")
    text = str(fields.get("text") or fields.get("description") or "")
    if not any((title, requirements, responsibilities, text)):
        raise ValueError("Vacancy source fields are empty")
    return {"title": title, "experience": experience, "requirements": requirements,
            "responsibilities": responsibilities, "text": text}


def extract_features(cv_text, vacancy):
    if not isinstance(cv_text, str) or not cv_text.strip():
        raise ValueError("A nonempty source CV is required")
    fields = source_vacancy(vacancy)
    cv = professional_text(cv_text)
    # Full professional source; annotation titles, scores, quotes and explanations
    # are never arguments to any signal extractor.
    vt = professional_text("\n".join(fields.values()))
    req = professional_text(fields["requirements"])
    title = professional_text(fields["title"])
    have, need = set(extract_skills(cv)), set(extract_skills(vt))
    cr, vr = roles(cv), roles(title)
    role = coverage(cr, vr)
    if not role and "software" in cr and vr & {"backend", "frontend", "fullstack"}:
        role = .5
    cl, vl = seniority(cv), seniority(title)
    level = 0.
    if vl and cl:
        # A range such as Middle/Senior accepts its lower bound.
        level = 1. if max(cl) >= min(vl) else -(min(vl) - max(cl)) / 4
    cy, vy = claimed_years(cv), required_years(fields["experience"])
    experience = 0.
    if vy == 0:
        experience = 1.  # No minimum is an allowance, not a demand for zero experience.
    elif vy is not None and cy is not None:
        experience = 1. if cy >= vy else -(vy - cy) / vy
    cd, vd = degree(cv), degree(req)
    education = 0.
    if cd is not None and vd is not None:
        education = 1. if cd >= vd else -(vd - cd) / 4
        if education > 0 and re.search(r"\bstudent\b|\bстудент(?:ка)?\b|in progress|обучаюсь", cv):
            education = .5
    languages_cv, languages_req = language_levels(cv), language_levels(req)
    language_scores = [1. if languages_cv[name] >= value else -(value - languages_cv[name]) / 6
                       for name, value in languages_req.items() if name in languages_cv]
    mandatory = set()
    for sentence in re.split(r"[;\n]|\.(?:\s|$)", req):
        if re.search(r"обязател|mandatory|required|must", sentence):
            mandatory.update(extract_skills(sentence))
    communication_needed = bool(re.search(r"communicat|коммуник|переговор|презентац", req))
    communication_have = bool(re.search(r"communicat|коммуник|переговор|презентац|presentations", cv))
    values = [coverage(have, need), len(have & need) / max(1, len(have | need)),
              coverage(detected(cv, TASKS), detected(vt, TASKS)), role, level, experience,
              education, coverage(detected(cv, EDUCATION_FIELDS), detected(req, EDUCATION_FIELDS)) if vd else 0.,
              sum(language_scores) / len(language_scores) if language_scores else 0.,
              coverage(have, mandatory), coverage(detected(cv, INDUSTRIES), detected(req, INDUSTRIES)),
              float(communication_needed and communication_have)]
    facts = {"confirmed_skill_matches": sorted(have & need),
             "skills_not_confirmed": sorted(need - have),
             "cv_professional_areas": sorted(cr), "vacancy_professional_areas": sorted(vr),
             "claimed_total_years": cy, "minimum_general_years": vy,
             "years_policy": "Only explicitly stated totals; job intervals are not inferred",
             "languages_compared": sorted(set(languages_cv) & set(languages_req)),
             "signals": dict(zip(FEATURE_NAMES, values))}
    return values, facts
