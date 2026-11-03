"""Local, source-grounded resume feedback and vacancy questions.

This module does not call a language model, train a model, save a conversation,
or add its generated answers to a candidate's professional evidence. Callers
must authorize access to the supplied candidate and vacancy before using it.
"""
from copy import deepcopy
import math
import re

from backend.matching import candidate_source
from ML2.features import (WORK_ACTION, claimed_years, positive_statement,
                          professional_text, required_skill_evidence,
                          source_vacancy, statements)
from NLP.skill_extractor import extract_skills


SOURCE_NOTE = "Ответ основан только на сохранённых данных вакансии и резюме; это не решение работодателя."
WORK_FORMAT = r"удал[её]н|remote|офис|office|гибрид|hybrid|работ\w*\s+из\s+дома|work(?:ing)?\s+from\s+home|on[\s-]?site"
WORK_LOAD = r"(?:полн|частичн|неполн)\w*\s+занятост|(?:part|full)[\s\-–—‑]*time|(?:полн|неполн)\w*\s+(?:рабоч\w*\s+)?д[её]н"
WORK_SCHEDULE = r"график|schedule|\b\d+\s*(?:час\w*|hours?)\b|\b[1-7]\s*/\s*[1-7]\b"


def _unique(values, limit=8):
    return list(dict.fromkeys(str(value) for value in values if value))[:limit]


def _number(value, digits=3, signed=False):
    value = float(value)
    if not math.isfinite(value):
        return "не рассчитан"
    if abs(value) < .5 * 10 ** -digits:
        value = 0.
    return ("+" if signed and value > 0 else "") + f"{value:.{digits}f}".replace(".", ",")


def _years(value):
    if value is None:
        return "не установлен"
    return f"{float(value):g} г."


def _vacancy_text(vacancy):
    return "\n".join(str(vacancy.get(key) or "") for key in
                     ("title", "requirements", "responsibilities", "description"))


def _required(vacancy):
    if not vacancy:
        return set()
    return set(required_skill_evidence(source_vacancy(vacancy)))


def _evidence(items, limit=8):
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not item.get("quote"):
            continue
        source = item.get("source", "resume")
        if source not in {"vacancy", "resume", "calculation", "clarification", "hr"}:
            continue
        quote = str(item["quote"])[:1200]
        if (source, quote) not in seen:
            result.append({"source": source, "quote": quote})
            seen.add((source, quote))
    return result[:limit]


def _vacancy_proof(vacancy, *fields):
    return [{"source": "vacancy", "quote": str(vacancy[field])}
            for field in fields if vacancy.get(field)]


def _breakdown(match):
    return (match.get("score_breakdown") or {}) if match else {}


def build_resume_feedback(candidate, vacancy=None, match=None):
    """Describe available evidence; never turn missing text into inability."""
    source, _, strong, weak = candidate_source(candidate)
    actions = [part for part in statements(source)
               if positive_statement(part) and WORK_ACTION.search(part)]
    strengths, gaps, recommendations = [], [], []
    if strong:
        strengths.append("В профессиональном тексте найдены навыки: " + ", ".join(sorted(strong)[:12]) + ".")
    else:
        gaps.append("Автоматически подтверждённые упоминания IT-навыков пока не найдены; проверьте извлечённый текст.")
        recommendations.append("Укажите технологии, с которыми действительно работали, и опишите их применение.")
    if actions:
        strengths.append("Есть описание выполненных действий или задач. Оно остаётся самоотчётом и требует проверки человеком.")
    else:
        gaps.append("Не найдены явные описания выполненных задач в распознаваемых формулировках.")
        recommendations.append("Добавьте 1–2 проекта: задача, ваша роль, использованные технологии и результат.")
    if weak:
        gaps.append("Требуют уточнения учебные или отрицательные формулировки: " + ", ".join(sorted(weak)[:8]) + ".")
        recommendations.append("Разделите практический опыт, знакомство с технологией и навыки, которые только изучаете.")
    if len(source) < 180:
        recommendations.append("Добавьте контекст работы: обязанности, проекты и конкретный личный вклад.")

    if vacancy is None:
        if claimed_years(source) is None:
            recommendations.append("Укажите длительность опыта по каждой роли явно; автоматический разбор не суммирует периоды работы по датам.")
        recommendations.append("Выберите вакансию, чтобы получить разбор нужных навыков, профильного стажа и балла ML2.")
        summary = "Проверено содержание сохранённого резюме. Соответствие конкретной вакансии пока не рассчитывалось."
    elif match is None:
        summary = f"Для «{vacancy['title']}» ещё нет расчёта сопоставления."
        recommendations.append("Запустите сравнение резюме с выбранной вакансией.")
    else:
        required = _required(vacancy)
        matched = set(match.get("matched_skills", []))
        uncertain = set(match.get("uncertain_skills", []))
        missing = (set(match.get("missing_skills", [])) | uncertain) & required
        confirmed_required = matched & required
        if confirmed_required:
            strengths.append("Из нужных навыков вакансии в тексте подтверждены: " + ", ".join(sorted(confirmed_required)) + ".")
        if missing:
            gaps.append("Нужные навыки без достаточных подтверждений в тексте: " + ", ".join(sorted(missing)) + ". Это не означает, что вы ими не владеете.")
            recommendations.append("Если такой опыт есть, опишите выполненную задачу с " + ", ".join(sorted(missing)[:4]) + ": свою роль и результат. Не добавляйте несуществующий опыт.")
        if not required:
            gaps.append("В названии и требованиях не распознаны нужные технологии; требования стоит проверить вручную.")
        details = _breakdown(match)
        experience = details.get("experience", {})
        relevant, minimum = experience.get("relevant_years"), experience.get("required_years")
        if relevant is not None:
            strengths.append("Из текста выделен профильный стаж: " + _years(relevant) + ". Это заявленные сведения, не независимая проверка.")
        elif minimum is not None and minimum > 0:
            gaps.append("Продолжительность опыта именно по выбранной роли автоматически не установлена.")
            recommendations.append("Уточните, сколько времени работали именно по профилю вакансии, отдельно от общего стажа.")
        if relevant is not None and minimum is not None and relevant < minimum:
            gaps.append(f"Установленный из текста профильный стаж {_years(relevant)} ниже минимума вакансии {_years(minimum)}.")
            recommendations.append("Проверьте правильность извлечённого стажа и отдельно обсудите с HR возможность рассмотрения при меньшем опыте.")
        for warning in details.get("warnings", []):
            gaps.append(warning)
        summary = (f"Для «{vacancy['title']}» балл ML2: {_number(match.get('score', 0), 2, True)}. "
                   f"Нужные навыки: {len(confirmed_required)} из {len(required)} распознанных подтверждены текстом. "
                   "Балл относительный и не является вероятностью найма.")
    result = {"summary": summary, "strengths": _unique(strengths), "gaps": _unique(gaps),
              "recommendations": _unique(recommendations), "source_note": SOURCE_NOTE}
    if match is not None:
        result["match"] = deepcopy(match)
    return result


def _score_answer(vacancy, candidate, match):
    if candidate is None or match is None:
        return {"topic": "score", "answer": "Выберите своё резюме, чтобы объяснить его балл именно для этой вакансии. По одной вакансии оценку кандидата рассчитать нельзя.", "evidence": []}
    details = _breakdown(match)
    parts = sorted(details.get("contributions", []), key=lambda row: -abs(float(row.get("contribution", 0))))
    changing = [row for row in parts if abs(float(row.get("contribution", 0))) > 1e-12]
    total = details.get("final_score", match.get("score", 0))
    lines = [f"Итоговый балл для «{vacancy['title']}»: {_number(total, 3, True)}. Это относительный результат сравнения, а не процент пригодности или решение о найме."]
    proof = []
    if changing:
        lines.append("Крупнейшие вклады: " + "; ".join(f"{row['label']}: {_number(row['contribution'], 3, True)}" for row in changing[:4]) + ".")
        for row in changing[:4]:
            proof.extend(row.get("evidence", []))
    else:
        lines.append("В текущем расчёте нет ненулевых вкладов профессиональных признаков. Это недостаток подтверждений в тексте, а не вывод об отсутствии способностей.")
    for rule in details.get("adjustments", []):
        lines.append(f"Отдельное правило «{rule['label']}»: {_number(rule['contribution'], 3, True)}. {rule['reason']}")
        proof.extend(rule.get("evidence", []))
    required = _required(vacancy)
    missing = (set(match.get("missing_skills", [])) | set(match.get("uncertain_skills", []))) & required
    if missing:
        lines.append("В тексте пока не подтверждены нужные навыки: " + ", ".join(sorted(missing)) + ".")
    if details.get("warnings"):
        lines.append(" ".join(details["warnings"][:2]))
    proof.insert(0, {"source": "calculation", "quote":
                    f"Модель: {details.get('model_version', 'не указана')}. "
                    f"Сумма признаков: {_number(details.get('raw_score', total), 6)}; "
                    f"сумма поправок: {_number(sum(row.get('contribution', 0) for row in details.get('adjustments', [])), 6)}; "
                    f"итог: {_number(total, 6)}."})
    return {"topic": "score", "answer": "\n\n".join(lines), "evidence": _evidence(proof),
            "score_breakdown": deepcopy(details)}


def answer_question(question, vacancy, candidate=None, match=None):
    """Answer supported intents from explicit fields and the supplied ML result."""
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise ValueError("Напишите вопрос о вакансии или расчёте балла.")
    if re.search(r"улучш|повыс|поднят|improv|increase|strengthen|добавить.*резюме|дописать|чего.*не хватает", normalized):
        if candidate is None:
            return {"topic": "improvement", "answer": "Выберите своё резюме, чтобы получить рекомендации по его описанию и требованиям этой вакансии.", "evidence": []}
        feedback = build_resume_feedback(candidate, vacancy, match)
        answer = "\n".join(feedback["recommendations"] or ["Дополните описание конкретных проектов, личного вклада и результата; не добавляйте несуществующий опыт."])
        return {"topic": "improvement", "answer": answer + "\n\nИзменение текста не гарантирует найм; HR проверяет сведения самостоятельно.",
                "evidence": _evidence(_vacancy_proof(vacancy, "requirements"))}
    if re.search(r"балл|скор|score|оценк|ранж|рейтинг|почему.*(?:ниже|выше|низк|высок)", normalized):
        return _score_answer(vacancy, candidate, match)
    if re.search(r"зарплат|оклад|оплат|salary|compensation|pay\b|жалақы", normalized):
        value = vacancy.get("salary")
        return {"topic": "salary", "answer": ("В вакансии указана оплата: " + str(value) + ". Условия и актуальность нужно уточнить у работодателя."
                                                   if value else "В сохранённой вакансии зарплата не указана. Я не могу определить её по должности или баллу; уточните у работодателя."),
                "evidence": _evidence(_vacancy_proof(vacancy, "salary"))}
    if re.search(r"город|локац|где.*работ|location|city|where.*(?:job|work)|қала", normalized):
        value = vacancy.get("city")
        return {"topic": "location", "answer": ("Указанный город: " + str(value) + ". Формат присутствия в офисе нужно проверить по описанию или у HR."
                                                     if value else "Город в сохранённой вакансии не указан. Уточните локацию у работодателя."),
                "evidence": _evidence(_vacancy_proof(vacancy, "city"))}
    if re.search(WORK_FORMAT + "|" + WORK_LOAD + r"|график|занятост|employment|schedule|формат.*работ", normalized):
        # Working hours and working location are different facts. Full-time is
        # not evidence of either remote availability or an office requirement.
        source_parts = statements(_vacancy_text(vacancy))
        structured = _vacancy_proof(vacancy, "employment")
        format_question = bool(re.search(WORK_FORMAT + r"|формат.*работ", normalized))
        load_question = bool(re.search(WORK_LOAD + r"|занятост|employment", normalized))
        schedule_question = bool(re.search(r"график|schedule", normalized))
        lines, fields = [], []
        if format_question:
            related = [item for item in structured if re.search(WORK_FORMAT, item["quote"], re.I)]
            related += [{"source": "vacancy", "quote": part} for part in source_parts if re.search(WORK_FORMAT, part, re.I)][:3]
            if related:
                lines.append("О формате работы в источнике указано: " + "; ".join(item["quote"] for item in related))
            else:
                lines.append("Формат работы в сохранённых данных не уточнён. Наличие удалённой работы по этим данным не подтверждено; уточните его у HR.")
            fields.extend(related)
        if load_question:
            related = structured + [{"source": "vacancy", "quote": part} for part in source_parts if re.search(WORK_LOAD, part, re.I)][:3]
            lines.append("О занятости в источнике указано: " + "; ".join(item["quote"] for item in related)
                         if related else "Вид занятости в сохранённых данных не указан. Уточните его у HR.")
            fields.extend(related)
        if schedule_question:
            related = [item for item in structured if re.search(WORK_SCHEDULE, item["quote"], re.I)]
            related += [{"source": "vacancy", "quote": part} for part in source_parts if re.search(WORK_SCHEDULE, part, re.I)][:3]
            lines.append("О графике в источнике указано: " + "; ".join(item["quote"] for item in related)
                         if related else "График и рабочие часы в сохранённых данных не уточнены. Уточните их у HR.")
            fields.extend(related)
        return {"topic": "employment", "answer": "\n\n".join(lines), "evidence": _evidence(fields)}
    if (re.search(r"язык|languages?|английск|русск|казахск|english|kazakh|қазақ|ағылшын", normalized)
            and not re.search(r"программир|programming", normalized)):
        snippets = [part for part in statements(_vacancy_text(vacancy))
                    if re.search(r"английск|русск|казахск|english|russian|kazakh|қазақ|ағылшын|\b[ABC][12]\b", part, re.I)]
        return {"topic": "languages", "answer": ("В описании языковых требований найдено:\n\n" + "\n\n".join(snippets[:3])
                                                     if snippets else "Отдельных требований к человеческим языкам в доступном описании не найдено. Уточните их у HR."),
                "evidence": _evidence([{"source": "vacancy", "quote": part} for part in snippets[:3]])}
    if re.search(r"образован|диплом|education|degree|бакалавр|магистр", normalized):
        snippets = [part for part in statements(_vacancy_text(vacancy))
                    if re.search(r"образован|диплом|education|degree|bachelor|master|бакалавр|магистр", part, re.I)]
        return {"topic": "education", "answer": ("В описании образования найдено:\n\n" + "\n\n".join(snippets[:3])
                                                     if snippets else "Отдельного требования к образованию в доступном описании не найдено. Это нужно уточнить у работодателя."),
                "evidence": _evidence([{"source": "vacancy", "quote": part} for part in snippets[:3]])}
    if re.search(r"стаж|опыт|experience|seniority|junior|middle|senior|сколько.*лет|years", normalized):
        required = str(vacancy.get("experience") or "")
        lines = ["Требование к опыту: " + required + "." if required else "Отдельное требование к длительности опыта в вакансии не указано."]
        proof = _vacancy_proof(vacancy, "experience")
        details = _breakdown(match)
        if candidate is not None and details:
            exp = details.get("experience", {})
            lines.append("Общий заявленный стаж: " + _years(exp.get("general_years")) + "; по этой роли: " + _years(exp.get("relevant_years")) + ".")
            lines.append(exp.get("reason", ""))
            proof.extend(exp.get("evidence", []))
        lines.append("Общий стаж другой профессии не переносится автоматически на эту вакансию; неизвестность не считается нулём.")
        return {"topic": "experience", "answer": "\n\n".join(value for value in lines if value), "evidence": _evidence(proof)}
    query_skills = extract_skills(normalized)
    if query_skills or re.search(r"требован|навык|стек|technolog|requirements|skills|stack|что.*(?:знать|уметь)|нужен|нужн[аоы]", normalized):
        required, all_skills = _required(vacancy), set(extract_skills(_vacancy_text(vacancy)))
        lines = []
        if query_skills:
            for skill in query_skills:
                if skill in required:
                    lines.append(f"{skill}: найден в названии или требованиях без отметки о желательности.")
                elif skill in all_skills:
                    lines.append(f"{skill}: упомянут в описании, но не выделен как нужный навык без пожеланий. Проверьте точную формулировку.")
                else:
                    lines.append(f"{skill}: в доступном описании не обнаружен. Это не доказывает, что работодатель не использует эту технологию.")
        elif required:
            lines.append("Распознаны нужные навыки без явно желательных: " + ", ".join(sorted(required)) + ".")
        if match is not None and candidate is not None:
            checked = set(query_skills) if query_skills else required
            confirmed = set(match.get("matched_skills", [])) & checked
            uncertain = (set(match.get("missing_skills", [])) | set(match.get("uncertain_skills", []))) & checked
            if confirmed:
                lines.append("В вашем тексте подтверждены упоминания: " + ", ".join(sorted(confirmed)) + ".")
            if uncertain:
                lines.append("В вашем тексте пока недостаточно подтверждений: " + ", ".join(sorted(uncertain)) + ". Это не оценка ваших реальных способностей.")
        if not lines:
            lines.append("Словарь не выделил конкретных технологий в требованиях. Проверьте исходное описание и уточните требования у HR.")
        return {"topic": "requirements", "answer": "\n\n".join(lines),
                "evidence": _evidence(_vacancy_proof(vacancy, "requirements", "responsibilities", "title"))}
    if re.search(r"кто.*работодател|какая\s+(?:это\s+)?компани|название\s+компани|which company|employer", normalized):
        company = vacancy.get("company")
        return {"topic": "company", "answer": "В вакансии указан работодатель: " + str(company) + "." if company else "Работодатель в сохранённых данных не указан.",
                "evidence": _evidence(_vacancy_proof(vacancy, "company"))}
    if re.search(r"(?:расскажи|опиши).*?(?:ваканси|позици)|tell.*(?:job|vacancy|role)|describe.*(?:job|vacancy|position)", normalized):
        facts = [str(vacancy.get("title") or "Название не указано")]
        for label, field in (("Компания", "company"), ("Город", "city"), ("Опыт", "experience")):
            if vacancy.get(field):
                facts.append(f"{label}: {vacancy[field]}")
        if vacancy.get("requirements"):
            facts.append("Требования из описания: " + str(vacancy["requirements"])[:1200])
        facts.append("Это сохранённое описание; актуальность открытой позиции нужно проверить у работодателя.")
        return {"topic": "overview", "answer": "\n\n".join(facts),
                "evidence": _evidence(_vacancy_proof(vacancy, "title", "company", "city", "experience", "requirements"))}
    # Retrieval only quotes the source. It does not synthesize absent policies,
    # company culture, interview outcomes or assertions of current job openness.
    ignored = {"какой", "какая", "какие", "почему", "можно", "вакансии", "вакансия", "работа", "what", "which", "about", "this", "that", "with", "есть"}
    tokens = {word for word in re.findall(r"[\w+#.]{4,}", normalized) if word not in ignored}
    snippets = [part for part in statements(_vacancy_text(vacancy))
                if tokens & set(re.findall(r"[\w+#.]{4,}", part.casefold()))]
    if snippets:
        return {"topic": "source", "answer": "В сохранённом описании есть такой фрагмент:\n\n" + "\n\n".join(snippets[:2]) + "\n\nДополнительные условия из него не следуют; уточните их у работодателя.",
                "evidence": _evidence([{"source": "vacancy", "quote": part} for part in snippets[:2]])}
    return {"topic": "unknown", "answer": "В сохранённых данных нет достаточного ответа на этот вопрос. Могу объяснить требования, указанный опыт, зарплату, город и расчёт балла по выбранному резюме. Другие условия нужно уточнить у HR.", "evidence": []}
