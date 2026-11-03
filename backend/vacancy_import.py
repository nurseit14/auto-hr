"""Conservative local vacancy-file preview; publication remains a separate step."""
from pathlib import Path
import re

from backend.documents import ResumeError, extract_resume, validate_text


FIELD_PATTERNS = {
    "title": r"название(?:\s+(?:вакансии|должности|позиции))?|должность|позиция|вакансия|job\s+title|position|role|лауазым",
    "company": r"компания|работодатель|company|employer|компания\s+атауы",
    "city": r"город|локация|location|city|қала",
    "experience": r"опыт(?:\s+работы)?|стаж|experience|тәжірибе",
    "salary": r"зарплата|оплата|оклад|salary|compensation|жалақы",
    "requirements": r"технические\s+требования|требования|requirements|required\s+skills|skills|біліктілік\s+талаптары",
    "description": r"обязанности|задачи|описание|responsibilities|description|duties|міндеттер",
}
LIMITS = {"title": 180, "company": 180, "city": 100, "experience": 180,
          "salary": 180, "requirements": 12000, "description": 12000}
IT_TITLE = re.compile(r"\bIT\b|\bИТ\b|software|developer|programmer|backend|frontend|full.?stack|devops|\bQA\b|\bSRE\b|data\s+(?:engineer|analyst|scientist)|machine\s+learning|разработ|программист|тестиров|аналитик|инженер|әзірлеуші", re.I)


def _vacancy_error(error):
    text = str(error).replace("резюме", "вакансии").replace("Резюме", "Описание вакансии")
    text = text.replace("опыт работы и навыки", "описание роли и требований")
    return ResumeError(text)


def parse_vacancy_preview(text, filename="vacancy.txt"):
    try:
        text = validate_text(text)
    except ResumeError as error:
        raise _vacancy_error(error) from error
    fields = {name: "" for name in FIELD_PATTERNS}
    blocks = {name: [] for name in FIELD_PATTERNS}
    current = None
    unlabelled = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        found = None
        for field, pattern in FIELD_PATTERNS.items():
            match = re.match(rf"^(?:{pattern})\s*[:：—–-]\s*(.*)$", line, re.I)
            if match:
                found = field
                value = match[1].strip()
                if value:
                    blocks[field].append(value)
                current = field if field in {"requirements", "description"} else None
                break
            if re.fullmatch(rf"(?:{pattern})\s*[:：]?", line, re.I):
                found = field
                current = field
                break
        if found is None:
            if current is not None:
                blocks[current].append(line)
                if current not in {"requirements", "description"}:
                    current = None
            else:
                unlabelled.append(line)
    for field, values in blocks.items():
        fields[field] = "\n".join(values).strip()
    if not fields["title"]:
        title = next((line for line in unlabelled if IT_TITLE.search(line) and len(line) <= LIMITS["title"]), "")
        fields["title"] = title
    warnings = ["Проверьте извлечённые поля перед публикацией. Файл не создаёт вакансию автоматически."]
    if not fields["requirements"]:
        fields["requirements"] = text
        warnings.append("Отдельный блок требований не найден: в поле требований сохранён исходный текст для ручной проверки.")
    if not fields["description"]:
        fields["description"] = text
    if not fields["company"]:
        warnings.append("Компания не указана явным полем. Заполните её вручную.")
    if not fields["city"]:
        warnings.append("Город не указан явным полем. Уточните его перед публикацией.")
    if not fields["title"]:
        warnings.append("Название IT-должности не выделено. Заполните его вручную.")
    for field, limit in LIMITS.items():
        if len(fields[field]) > limit:
            fields[field] = fields[field][:limit]
            warnings.append(f"Поле «{field}» сокращено до {limit} символов. Проверьте конец текста.")
    return {"fields": fields, "filename": Path(filename or "vacancy").name[:200], "warnings": warnings}


def preview_vacancy(data, filename):
    try:
        text = extract_resume(data, filename)
    except ResumeError as error:
        raise _vacancy_error(error) from error
    return parse_vacancy_preview(text, filename)
