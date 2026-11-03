"""Fixed professional signals extracted from source documents, never annotations.

This first feature model has no free-text vocabulary or personal-identity slots.
Unknown evidence remains unknown; missing technologies are not proof of inability.
"""

import hashlib
import math
from pathlib import Path
import re

from NLP.skill_extractor import extract_skills
from NLP.skills_dictionary import SKILL_ALIASES


FEATURE_VERSION = "professional-signals-v3"
FEATURE_NAMES = ["skill_coverage", "skill_jaccard", "task_coverage", "role_compatibility",
                 "seniority_compatibility", "experience_compatibility",
                 "education_level_compatibility", "education_field_coverage",
                 "language_compatibility", "mandatory_skill_confirmation",
                 "industry_coverage", "communication_confirmation",
                 "required_skill_coverage", "demonstrated_skill_coverage",
                 "relevant_experience_fulfillment"]


def feature_fingerprint():
    root = Path(__file__).resolve().parent.parent
    source = b"".join((root / path).read_bytes() for path in
                      ("ML2/features.py", "NLP/skill_extractor.py", "NLP/skills_dictionary.py"))
    return hashlib.sha256(source).hexdigest()

ROLES = {
    "backend": r"backend|back.end|б[эе]к[эе]нд|(?:java|python|c#|\.net|golang|node\.?js)\s+(?:developer|разработчик)",
    "frontend": r"front[ ._-]?end|фронт.?енд|фронт.?энд",
    "fullstack": r"full[ -]?stack|фул[ -]?ст[еэ]к",
    "data_engineering": r"data\s+engineer(?:ing)?|инженер\w*\s+(?:по\s+)?данн|(?:etl|dwh)\s+(?:developer|engineer)|разработчик\w*\s+(?:etl|dwh)",
    "data_analysis": r"data\s+analyst|аналитик\w*\s+данных|bi\s+(?:analyst|developer)|business intelligence",
    "data_science": r"data\s+scien(?:ce|tist)|machine learning|ml\s+(?:engineer|research)",
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
    "data_pipelines": r"\betl\b|\belt\b|data\s+pipelines?|data\s+warehouse|\bdwh\b|поток\w*\s+данных|трансформац\w*\s+данных|хранилищ\w*\s+данных",
    "backend_systems": r"backend\s+(?:systems?|services?)|microservices?|микросервис|серверн\w*\s+(?:систем|сервис)|б[эе]кенд\s+сервис",
    "frontend_interfaces": r"front.end\s+(?:interfaces?|applications?)|user\s+interfaces?|клиентск\w*\s+интерфейс|пользовательск\w*\s+интерфейс",
    "ml_models": r"train\w*\s+(?:ml\s+)?models?|model\s+training|обучени\w*\s+модел|нейронн\w*\s+сет|neural\s+networks?",
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
# These common tools can confirm a skill, but cannot establish the duration of
# a specialised job. For example, Git/REST API alone are not Data Engineering.
COMMON_SKILLS = {"git", "github", "gitlab", "rest api", "agile", "scrum", "jira", "confluence"}
SPECIALISED_TASKS = {"data_pipelines", "backend_systems", "frontend_interfaces", "ml_models", "networks"}
# Task evidence can establish a professional area even when a CV uses a
# different job title. It does not establish years or a seniority by itself.
TASK_ROLES = {"data_pipelines": "data_engineering", "backend_systems": "backend",
              "frontend_interfaces": "frontend", "ml_models": "data_science",
              "networks": "telecom"}
OPTIONAL_REQUIREMENT = re.compile(r"желател\w*|будет\s+плюсом|преимуществ\w*|необязател\w*|"
                                  r"\boptional\b|\bpreferred\b|\bnice[ -]to[ -]have\b|\ba plus\b", re.I)
WORK_ACTION = re.compile(r"\b(?:built|build|developed|implemented|created|maintained|deployed|"
                         r"designed|optimized|integrated|automated|tested|delivered|migrated|used)\b|"
                         r"разработ\w*|созда\w*|внедр\w*|поддержива\w*|настро\w*|разверн\w*|"
                         r"разв[её]рн\w*|оптимизир\w*|автоматизир\w*|интегрир\w*|тестирова\w*", re.I)
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


def statements(text):
    """Small source spans, retaining context instead of treating a CV as a bag."""
    return [part.strip() for part in re.split(
        r"\n+|(?<=[.!?;])\s+|,?\s+(?:но|однако|but|however)\s+|"
        r",?\s+and\s+(?=(?:(?:i|we)\s+)?(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't|never)\s+(?:use|used|work|worked|build|built))",
        text, flags=re.I) if part.strip()]


def required_skill_evidence(fields):
    """Keep requirements distinct from explicit wishes and generic source text.

    Absence of a skill in a CV remains unknown. This selects what a vacancy
    asks for; it does not turn missing requirements into negative labels.
    """
    output = {}
    title = professional_text(fields["title"])
    for skill in extract_skills(title):
        output.setdefault(skill, []).append({"quote": title[:400], "source": "vacancy"})
    optional_section = False
    for part in statements(professional_text(fields["requirements"])):
        if re.match(r"(?:обязател\w*|требован\w*|must[ -]have|required)\s*[:：]", part, re.I):
            optional_section = False
        optional = OPTIONAL_REQUIREMENT.search(part)
        if optional:
            # "Python, SQL; желательно Docker" must retain Python and SQL.
            required_part = part[:optional.start()].rstrip(" ,;:—–-")
            trailing = part[optional.end():].strip(" ().,;:—–-")
            if required_part and not trailing:
                # In "Docker (optional)" the modifier describes the skill
                # before it, unlike prefix "желательно Docker".
                prior = required_part.rsplit(",", 1)[0] if "," in required_part else ""
                required_part = prior if re.search(r"\brequired\b|\bmust\b|обязател\w*", prior, re.I) else ""
            if not optional_section:
                for skill in extract_skills(required_part):
                    output.setdefault(skill, []).append({"quote": required_part[:400], "source": "vacancy"})
            optional_section = not required_part and bool(re.search(r"[:：]\s*$", part))
            continue
        if optional_section:
            continue
        for skill in extract_skills(part):
            output.setdefault(skill, []).append({"quote": part[:400], "source": "vacancy"})
    if not output and not fields["requirements"].strip():
        # Some datasets have no structured requirements. Fall back only then;
        # an explicit optional-only section must not become mandatory again.
        for field in ("responsibilities", "text"):
            for part in statements(professional_text(fields[field])):
                if OPTIONAL_REQUIREMENT.search(part):
                    continue
                for skill in extract_skills(part):
                    output.setdefault(skill, []).append({"quote": part[:400], "source": "vacancy"})
    return output


def role_coverage(candidate_roles, required_roles):
    """Fullstack explicitly includes both web layers; other roles stay distinct."""
    candidate = set(candidate_roles)
    required = set(required_roles)
    if "fullstack" in candidate:
        candidate.update(("backend", "frontend"))
    if "fullstack" in required:
        required.discard("fullstack")
        required.update(("backend", "frontend"))
    if candidate & {"backend", "frontend", "fullstack"}:
        candidate.add("software")
    return coverage(candidate, required)


def uncertain_skill_mention(text, skill):
    """A shared weak/negative list predicate governs only its skill mentions."""
    weak_before = r"(?:\b(?:learning|learn|beginner(?: in)?|basic(?: knowledge of)?|no experience(?: with| in)?|never used)\b|изучаю|изучить|осваиваю|освоить|базов\w*(?: знани\w*)?|не работал(?: с| в)?|не использовал|нет опыта(?: работы)?(?: с| в)?|не знаю|знаком(?:ство)? с)\s*$|(?:(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't|never)\s+(?:use[ds]?|work(?:ed)?(?:\s+(?:with|in))?)|never (?:built|created|developed)|не (?:строил|разрабатывал|создавал|делал|занимался))[^.!?;]{0,100}$"
    weak_after = r"^\s*[:—–-]?\s*(?:только\s+)?(?:изучаю|осваиваю|не работал|не использовал|не знаю|нет опыта|\b(?:beginner|basic|learning|no experience|never used)\b)"
    denied_after = re.compile(
        r"^\s*[:—–-]?\s*(?:(?:is|was)\s+(?:a[n]?\s+)?(?:tool|technology|framework|language|database)\s+(?:(?:that|which)\s+)?)?"
        r"(?:(?:i|we|he|she)\s+)?(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't|never)\s+(?:use[ds]?|work(?:ed)?|built|created|developed)\b|"
        r"^\s*[:—–-]?\s*(?:(?:опыт\w*|практик\w*)[^.!?;]{0,50})?\bнет\b", re.I)
    flags = []
    for alias in {skill, *SKILL_ALIASES.get(skill, [])}:
        for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text, flags=re.I):
            before, after = text[:match.start()], text[match.end():]
            # Check an entity's direct relative clause before stripping other
            # technologies: "used Python and have not used Docker" must not
            # transfer Docker's denial back to Python.
            direct_denial = bool(denied_after.search(after))
            if re.search(r"(?:не знаю|не работал|не использовал|изучаю|осваиваю|never used|no experience)\s*,\s*$", before, re.I):
                before = before.rsplit(",", 1)[-1]
            for other in set(extract_skills(before + " " + after)) - {skill}:
                for name in sorted({other, *SKILL_ALIASES.get(other, [])}, key=len, reverse=True):
                    before = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", before, flags=re.I)
                    after = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", after, flags=re.I)
            before = re.sub(r"(?:\s|[,/&]|\b(?:and|or|и|или)\b)+$", " ", before, flags=re.I)
            after = re.sub(r"^(?:\s|[,/&]|\b(?:and|or|и|или)\b)+", " ", after, flags=re.I)
            flags.append(bool(direct_denial or re.search(weak_before, before, re.I) or re.search(weak_after, after, re.I)))
    return bool(flags) and all(flags)


def skill_evidence(text):
    result = {}
    for part in statements(text):
        for skill in extract_skills(part):
            if not uncertain_skill_mention(part, skill):
                result.setdefault(skill, []).append({"quote": part[:400], "source": "resume"})
    return result


def positive_statement(part):
    return not re.search(
        r"не работал|нет опыта|опыт\w*[^.!?;]{0,100}\bнет\b|never worked|no experience|never (?:built|used|created|developed)|"
        r"\b(?:(?:have|has|had|did|do)\s+not|haven't|hasn't|hadn't|didn't|don't)\s+(?:use[ds]?|work(?:ed)?|build|built|create[ds]?|develop(?:ed)?)\b|"
        r"не (?:строил|разрабатывал|создавал|делал|занимался)|not (?:a |an )?(?:data|backend|frontend)|"
        r"хочу стать|планирую стать|хочу изучить|планирую изучить|хочу освоить|изучаю|осваиваю|"
        r"aspiring|want to (?:be|become|learn)|looking to (?:become|learn)|plan to learn|"
        r"^\s*(?:currently\s+)?learning\b|\bi am learning\b|\bcurrently learning\b", part, re.I)


def positive_roles(text):
    """Negative career statements do not establish a professional profile."""
    output = set()
    for part in statements(text):
        if not positive_statement(part):
            continue
        output.update(roles(part))
    if len(output) > 1:
        output.discard("software")
    return output


DURATION = re.compile(r"(?P<years>\d+(?:[.,]\d+)?)\s*(?P<unit>years?|лет|года?|months?|месяц\w*)(?:\s+(?P<months>\d+)\s*(?:месяц\w*|months?))?", re.I)


def role_mentions(text):
    return [(match.start(), match.end(), name) for name, pattern in ROLES.items()
            if name != "software" for match in re.finditer(
                rf"(?:{pattern})(?:\s+(?:developer|engineer|разработчик|инженер))?", text, re.I)]


def distance(span, other):
    return max(0, other[0] - span[1], span[0] - other[1])


def relevant_experience(text, required_roles, candidate_roles, supported):
    """Explicit role-scoped years; a mixed-career total is never allocated.

    A total can be associated with a single supported professional profile. It
    remains a claimed total, not inferred dated employment or independently
    verified experience. Multiple claims are not summed because they may overlap.
    """
    total = claimed_years(text)
    if not supported:
        return None, "Стаж не учтён: нет подтверждений работы по профилю вакансии и её техническим требованиям.", []
    claims = []
    for part in statements(text):
        if not positive_statement(part):
            continue
        mentions = role_mentions(part)
        for match in DURATION.finditer(part):
            # A plain overall total does not inherit the next role in a list.
            prefix = part[max(0, match.start() - 45):match.start()]
            suffix = part[match.end():match.end() + 55]
            if (re.search(r"(?:общ(?:ий|его)\s+(?:опыт|стаж)|(?:total|overall)\s+(?:work\s+)?experience)\s*[:—–-]?\s*$", prefix, re.I)
                    or re.search(r"^\s*(?:(?:of\s+)?(?:total|overall)\b|(?:of\s+)?experience\s*,?\s*including|(?:общего\s+)?стажа\s*,?\s*из них)", suffix, re.I)):
                continue
            if (re.search(r"(?:total|общий|опыт работы|work experience)\s*[:—–-]?\s*$", prefix, re.I)
                    and not (len(list(DURATION.finditer(part))) == 1 and mentions)):
                continue
            nearest = sorted((distance(match.span(), (start, end)), role) for start, end, role in mentions)
            if not nearest or nearest[0][0] > 60:
                continue
            closest = {role for gap, role in nearest if gap == nearest[0][0]}
            if not closest or not closest <= required_roles:
                continue
            years = float(match["years"].replace(",", "."))
            if re.match(r"months?|месяц", match["unit"], re.I):
                years /= 12
            years += float(match["months"] or 0) / 12
            if not math.isfinite(years) or not 0 <= years <= 100:
                continue
            claims.append((years, {"quote": part[:400], "source": "resume"}))
    if claims:
        years, quote = max(claims, key=lambda claim: claim[0])
        return years, "Учтён явно указанный стаж по профилю вакансии; отдельные периоды не суммировались.", [quote]
    specific = candidate_roles - {"software"}
    if total is not None and required_roles and specific and specific <= required_roles:
        quotes = [{"quote": part[:400], "source": "resume"} for part in statements(text)
                  if claimed_years(part) is not None or positive_roles(part) & required_roles]
        return total, "Общий заявленный стаж отнесён к единственному подтверждённому профилю резюме; это самоотчёт, а не расчёт по датам.", quotes[:3]
    return None, "Релевантный стаж не установлен: общий стаж смешанного или неуточнённого профиля не переносится на эту вакансию.", []


def relevant_seniority(text, required_roles, candidate_roles):
    levels, evidence = [], []
    for part in statements(text):
        if not (positive_roles(part) & required_roles):
            continue
        mentions = role_mentions(part)
        found = []
        for value, pattern in SENIORITY_PATTERNS:
            for match in re.finditer(pattern, part, re.I):
                nearest = sorted((distance(match.span(), (start, end)), role) for start, end, role in mentions)
                closest = {role for gap, role in nearest if gap == nearest[0][0]} if nearest else set()
                if closest and closest <= required_roles:
                    found.append(value)
        if found:
            levels.extend(found)
            evidence.append({"quote": part[:400], "source": "resume"})
    if not levels and candidate_roles and candidate_roles <= required_roles:
        levels = seniority(text)
    return levels, evidence


SENIORITY_PATTERNS = [(0, r"\bintern\b|internship|стаж[её]р"),
                           (1, r"\bjunior\b|\bjun\b|младший|джун"),
                           (2, r"\bmiddle\b|\bmid.level\b|мидл"),
                           (3, r"\bsenior\b|\bsr\b|старший|сеньор"),
                           (4, r"\b(?:team|tech|technical)\s+lead\b|\blead\s+(?:developer|engineer|designer)\b|\bhead of\b|ведущий|руководитель")]


def seniority(text):
    found = []
    for level, pattern in SENIORITY_PATTERNS:
        if re.search(pattern, text, re.I):
            found.append(level)
    return found


def claimed_years(text):
    # Only a claimed total; student dates and partially dated job histories do not
    # become guessed employment durations. No overlapping dates are summed.
    m = re.search(r"(?:опыт работы|work experience)\s*[:—–-]?\s*(\d+(?:[.,]\d+)?)\s*"
                  r"(?:лет|год\w*|years?)(?:\s+(\d+)\s*(?:месяц\w*|months?))?", text, re.I)
    if m:
        years = float(m[1].replace(",", ".")) + float(m[2] or 0) / 12
        return years if math.isfinite(years) and 0 <= years <= 100 else None
    m = re.search(r"(?:with|over)\s+(\d+(?:\.\d+)?)\+?\s+years?\s+(?:of\s+)?experience", text, re.I)
    years = float(m[1]) if m else None
    return years if years is not None and math.isfinite(years) and 0 <= years <= 100 else None


def required_years(text):
    if re.search(r"нет опыта|без опыта|no experience", text, re.I):
        return 0.
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    years = float(m[0].replace(",", ".")) if m else None
    return years if years is not None and math.isfinite(years) and 0 <= years <= 100 else None


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
    evidence = skill_evidence(cv)
    have, need = set(evidence), set(extract_skills(vt))
    required_evidence = required_skill_evidence(fields)
    required_skills = set(required_evidence)
    cr, vr = positive_roles(cv), roles(title)
    core_need = (required_skills or need) - COMMON_SKILLS
    core_coverage = coverage(have, core_need)
    positive_cv = "\n".join(part for part in statements(cv) if positive_statement(part))
    cv_tasks, vacancy_tasks = detected(positive_cv, TASKS), detected(vt, TASKS)
    specialised_tasks = vacancy_tasks & SPECIALISED_TASKS
    specific_tasks = coverage(cv_tasks, specialised_tasks)
    task_roles = {TASK_ROLES[name] for name in cv_tasks & specialised_tasks}
    raw_role = role_coverage(cr | task_roles, vr)
    support = max(core_coverage, specific_tasks)
    # Titles, generic years and ubiquitous tools cannot manufacture relevance.
    # Technical evidence changes the experience/level signal continuously too.
    supported = bool(raw_role > 0 and support > 0)
    role = raw_role * support
    role_reason = ("Профиль вакансии подтверждён названием совместимой роли или профильными задачами вместе с техническими свидетельствами."
                   if supported else "Профиль вакансии не подтверждён одновременно ролью и связанными технологиями или задачами.")
    scoped_roles = set(vr)
    if vr and vr <= {"backend", "frontend"}:
        scoped_roles.add("fullstack")
    elif vr == {"software"}:
        scoped_roles.update(("backend", "frontend", "fullstack"))
    cl, level_evidence = relevant_seniority(cv, scoped_roles, cr)
    vl = seniority(title)
    level = 0.
    if supported and vl and cl:
        # A range such as Middle/Senior accepts its lower bound.
        level = (1. if max(cl) >= min(vl) else -(min(vl) - max(cl)) / 4) * role
    cy, vy = claimed_years(cv), required_years(fields["experience"])
    ry, experience_reason, experience_evidence = relevant_experience(cv, scoped_roles, cr, supported)
    experience = 0.
    if vy is None:
        experience_reason = "Вакансия не указывает минимальный стаж; длительность показана для проверки и не повышает этот признак."
    elif supported and vy == 0:
        experience_reason = "Вакансия не требует стажа: отсутствие минимума не даёт дополнительных баллов."
    elif supported and vy is not None and ry is not None:
        experience = (1. if ry >= vy else -(vy - ry) / vy) * role
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
    # Supporting education/language/soft evidence is useful within a related
    # professional context; it must not outrank a missing technical profile.
    context = max(role, core_coverage, specific_tasks)
    task_score = coverage(cv_tasks, vacancy_tasks) * context
    demonstrated = {}
    for part in statements(cv):
        if not positive_statement(part) or not WORK_ACTION.search(part):
            continue
        for skill in set(extract_skills(part)) & have & required_skills:
            if not uncertain_skill_mention(part, skill):
                demonstrated.setdefault(skill, []).append({"quote": part[:400], "source": "resume"})
    fulfillment = (min(1., ry / vy) * role
                   if supported and ry is not None and vy is not None and vy > 0 else 0.)
    values = [coverage(have, need), len(have & need) / max(1, len(have | need)),
              task_score, role, level, experience,
              education * context,
              coverage(detected(cv, EDUCATION_FIELDS), detected(req, EDUCATION_FIELDS)) * context if vd else 0.,
              (sum(language_scores) / len(language_scores)) * context if language_scores else 0.,
              coverage(have, mandatory), coverage(detected(cv, INDUSTRIES), detected(req, INDUSTRIES)) * context,
              float(communication_needed and communication_have) * context,
              coverage(have, required_skills), coverage(set(demonstrated), required_skills),
              fulfillment]
    matched = sorted(have & need)
    matched_evidence = [quote for skill in matched for quote in evidence[skill]][:4]
    role_evidence = [{"quote": part[:400], "source": "resume"} for part in statements(cv)
                     if positive_roles(part) & vr][:3]
    if task_roles & vr:
        role_evidence += [{"quote": part[:400], "source": "resume"} for part in statements(cv)
                          if positive_statement(part) and detected(part, TASKS) & specialised_tasks][:2]
    reasons = {
        "skill_coverage": f"Подтверждено {len(matched)} из {len(need)} обнаруженных навыков вакансии. Отсутствие упоминания — неизвестность.",
        "skill_jaccard": "Совпавшие навыки сопоставлены с объединением навыков резюме и вакансии.",
        "task_coverage": "Учитываются совпавшие задачи в профессиональном контексте; общие слова о проектах сами по себе не подтверждают специализацию.",
        "role_compatibility": role_reason,
        "seniority_compatibility": ("Уровень сравнивается только по подтверждённому профилю и с учётом технических свидетельств."
                                    if supported and cl and vl else "Уровень не учтён: нет подтверждённого профиля или явно сопоставимых уровней."),
        "experience_compatibility": experience_reason,
        "education_level_compatibility": "Сопоставляются только явно указанные уровни образования в связанном профессиональном контексте.",
        "education_field_coverage": "Сопоставляется явно указанное направление образования.",
        "language_compatibility": "Языки сравниваются только при явно указанном уровне с обеих сторон и профессиональных подтверждениях.",
        "mandatory_skill_confirmation": f"Подтверждено {len(have & mandatory)} из {len(mandatory)} распознанных явно обязательных навыков.",
        "industry_coverage": "Совпадение явно названной отрасли учитывается в связанном профессиональном контексте.",
        "communication_confirmation": "Коммуникация учитывается, когда она требуется вакансией и упомянута в связанном профессиональном контексте.",
        "required_skill_coverage": f"Подтверждено {len(have & required_skills)} из {len(required_skills)} распознанных навыков названия и требований; явно желательные навыки исключены.",
        "demonstrated_skill_coverage": f"Для {len(demonstrated)} из {len(required_skills)} нужных навыков найдены описания выполненных задач; простой список навыков не заменяет такое описание.",
        "relevant_experience_fulfillment": (f"Подтверждено {ry:g} профильных лет при минимуме {vy:g}; выполнение минимума учитывается вместе с профессиональными подтверждениями."
                                            if supported and ry is not None and vy is not None and vy > 0 else
                                            "Выполнение профильного стажа неизвестно или минимум не задан; неизвестность не означает нулевой стаж."),
    }
    feature_evidence = {name: [] for name in FEATURE_NAMES}
    for name in ("skill_coverage", "skill_jaccard"):
        feature_evidence[name] = matched_evidence
    feature_evidence["mandatory_skill_confirmation"] = [quote for skill in sorted(have & mandatory) for quote in evidence[skill]][:4]
    feature_evidence["role_compatibility"] = role_evidence + matched_evidence[:2]
    feature_evidence["seniority_compatibility"] = level_evidence if supported else []
    feature_evidence["experience_compatibility"] = experience_evidence
    feature_evidence["required_skill_coverage"] = [quote for skill in sorted(have & required_skills) for quote in evidence[skill]][:4]
    feature_evidence["demonstrated_skill_coverage"] = [quote for skill in sorted(demonstrated) for quote in demonstrated[skill]][:4]
    feature_evidence["relevant_experience_fulfillment"] = experience_evidence
    for name, patterns in (("task_coverage", TASKS), ("industry_coverage", INDUSTRIES), ("education_field_coverage", EDUCATION_FIELDS)):
        feature_evidence[name] = [{"quote": part[:400], "source": "resume"} for part in statements(cv)
                                  if positive_statement(part) and detected(part, patterns) & detected(vt, patterns)][:3]
    for name, predicate in (("education_level_compatibility", lambda part: degree(part) is not None),
                            ("language_compatibility", lambda part: bool(language_levels(part))),
                            ("communication_confirmation", lambda part: bool(re.search(r"communicat|коммуник|переговор|презентац", part, re.I)))):
        feature_evidence[name] = [{"quote": part[:400], "source": "resume"} for part in statements(cv) if predicate(part)][:3]
    warnings = []
    if not supported:
        warnings.append("Релевантный профессиональный профиль не подтверждён; общий стаж и название уровня не повышают балл.")
    if core_need and not (have & core_need):
        warnings.append("Ключевые обнаруженные технологии вакансии не подтверждены; уточните опыт кандидата.")
    if vy is not None and vy > 0 and ry is None:
        warnings.append("Продолжительность опыта именно по этой вакансии не установлена.")
    facts = {"confirmed_skill_matches": sorted(have & need),
             "skills_not_confirmed": sorted(need - have),
             "cv_professional_areas": sorted(cr), "vacancy_professional_areas": sorted(vr),
             "claimed_total_years": cy, "relevant_years": ry, "minimum_general_years": vy,
             "years_policy": "Explicit role-scoped durations, or a claimed total in a single supported professional profile; mixed totals and dated intervals are not allocated or summed",
             "experience_reason": experience_reason, "experience_evidence": experience_evidence,
             "role_reason": role_reason, "relevance_supported": supported,
             "technical_support": support, "core_skill_coverage": core_coverage,
             "required_skills": sorted(required_skills),
             "demonstrated_skills": sorted(demonstrated),
             "feature_reasons": reasons, "feature_evidence": feature_evidence, "warnings": warnings,
             "languages_compared": sorted(set(languages_cv) & set(languages_req)),
             "signals": dict(zip(FEATURE_NAMES, values))}
    return values, facts
