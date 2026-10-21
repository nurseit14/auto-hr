"""Public fictional inputs, including the user's scoring example."""

from ML2.data import digest, make_vacancy


def demo_inputs():
    text = "Junior Java разработчик. Опыт работы 1 год. Навыки: Java, Spring, SQL."
    cvs = [{"cv_id": "demo_java_junior", "text": text, "skills": ["java", "spring", "sql"],
            "text_sha256": digest(text)}]
    rows = [
        {"vacancy_id": "demo_middle", "title": "Middle Java разработчик", "experience": "1–3 года",
         "requirements": "Java, Spring, SQL, Python", "skills": "java, spring, sql, python"},
        {"vacancy_id": "demo_junior", "title": "Junior Java разработчик", "experience": "От 1 года",
         "requirements": "Java, Spring, SQL", "skills": "java, spring, sql"},
        {"vacancy_id": "demo_senior", "title": "Senior Java разработчик", "experience": "От 5 лет",
         "requirements": "Java, Spring, SQL, Kafka", "skills": "java, spring, sql, kafka"},
        {"vacancy_id": "demo_frontend", "title": "Middle Frontend разработчик", "experience": "От 2 лет",
         "requirements": "JavaScript, React, TypeScript", "skills": "javascript, react, typescript"},
        {"vacancy_id": "demo_python", "title": "Junior Python разработчик", "experience": "Без опыта",
         "requirements": "Python, Django, SQL", "skills": "python, django, sql"},
    ]
    return cvs, [make_vacancy(row) for row in rows]
