"""Vacancy previews preserve source, require review and use bounded extraction."""
from io import BytesIO
import zipfile

import pytest
from pypdf import PdfWriter

from backend.documents import MAX_FILE_BYTES, ResumeError
from backend.vacancy_import import parse_vacancy_preview, preview_vacancy


TEXT = """Должность: Middle Python Backend Developer
Компания: Example IT Team
Город: Алматы
Опыт работы: От 3 лет
Зарплата: 500 000–800 000 ₸
Требования:
Python, FastAPI, PostgreSQL.
Желательно Docker.
Обязанности:
Разрабатывать REST API и backend сервисы.
"""


def test_labelled_fields_are_extracted_without_publishing():
    result = parse_vacancy_preview(TEXT, "/temporary/vacancy.txt")
    assert result["fields"] == {
        "title": "Middle Python Backend Developer", "company": "Example IT Team", "city": "Алматы",
        "experience": "От 3 лет", "salary": "500 000–800 000 ₸",
        "requirements": "Python, FastAPI, PostgreSQL.\nЖелательно Docker.",
        "description": "Разрабатывать REST API и backend сервисы.",
    }
    assert result["filename"] == "vacancy.txt"
    assert "не создаёт вакансию автоматически" in result["warnings"][0]
    assert "id" not in result


def test_unknown_company_city_salary_remain_unknown_and_source_is_retained():
    source = "Python Backend Developer\nBuild Python FastAPI services with PostgreSQL."
    result = parse_vacancy_preview(source)
    assert result["fields"]["title"] == "Python Backend Developer"
    assert result["fields"]["company"] == ""
    assert result["fields"]["city"] == ""
    assert result["fields"]["salary"] == ""
    assert result["fields"]["requirements"] == source
    assert result["fields"]["description"] == source
    assert any("Компания не указана" in warning for warning in result["warnings"])


@pytest.mark.parametrize("label", ["Название", "Название вакансии", "Название должности", "Название позиции"])
def test_russian_title_labels_are_removed_from_extracted_role(label):
    result = preview_vacancy(TEXT.replace("Должность", label, 1).encode(), "vacancy.txt")
    assert result["fields"]["title"] == "Middle Python Backend Developer"
    assert result["fields"]["company"] == "Example IT Team"
    assert result["fields"]["requirements"] == "Python, FastAPI, PostgreSQL.\nЖелательно Docker."


def test_txt_and_real_docx_extraction_have_the_same_labelled_preview():
    txt = preview_vacancy(b"\xef\xbb\xbf" + TEXT.encode(), "vacancy.TXT")
    buffer = BytesIO()
    from xml.sax.saxutils import escape
    paragraphs = "".join(f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>" for line in TEXT.splitlines())
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + paragraphs + '</w:body></w:document>')
    docx = preview_vacancy(buffer.getvalue(), "vacancy.docx")
    assert txt["fields"] == docx["fields"]


@pytest.mark.parametrize("data,filename", [
    (b"", "vacancy.txt"),
    (b"\xff\xfe\x00", "vacancy.txt"),
    (b"Not a valid document", "vacancy.exe"),
    (b"not a pdf", "vacancy.pdf"),
    (b"not a zip", "vacancy.docx"),
])
def test_invalid_files_fail_with_vacancy_copy(data, filename):
    with pytest.raises(ResumeError) as reason:
        preview_vacancy(data, filename)
    assert "резюме" not in str(reason.value).casefold()


def test_file_size_and_scanned_pdf_do_not_bypass_extraction_limits():
    with pytest.raises(ResumeError, match="8 МБ"):
        preview_vacancy(b"x" * (MAX_FILE_BYTES + 1), "vacancy.txt")
    pdf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(pdf)
    with pytest.raises(ResumeError, match="скан"):
        preview_vacancy(pdf.getvalue(), "vacancy.pdf")


def test_preview_truncation_is_visible_and_within_create_field_limits():
    source = "Job title: Python Developer\nCompany: " + "A" * 200 + "\nRequirements:\n" + "Python backend services. " * 650
    result = parse_vacancy_preview(source)
    assert len(result["fields"]["requirements"]) == 12000
    assert len(result["fields"]["company"]) == 180
    assert any("сокращено" in warning for warning in result["warnings"])


def test_unlabelled_non_role_does_not_invent_a_position():
    result = parse_vacancy_preview("Интересная работа в дружной команде. Условия обсуждаются на встрече.")
    assert result["fields"]["title"] == ""
    assert any("должности не выделено" in warning for warning in result["warnings"])
