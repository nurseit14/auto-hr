"""Bounded local resume extraction; uploaded source files are not retained."""
from io import BytesIO
from pathlib import PurePath
import re
from zipfile import ZipFile, BadZipFile
from xml.etree import ElementTree

from pypdf import PdfReader, apply_configuration

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 60_000


class ResumeError(ValueError):
    pass


def _pdf_text(data):
    # Bound decoded streams inside the decompressor, before extraction allocates
    # them. Context-local limits also apply when extraction runs in a worker.
    with apply_configuration(maximum_declared_stream_length=12_000_000,
                             zlib_maximum_output_length=12_000_000,
                             lzw_maximum_output_length=12_000_000,
                             run_length_maximum_output_length=12_000_000,
                             array_based_stream_maximum_output_length=12_000_000,
                             stream_decoding_work_maximum_length=24_000_000,
                             page_tree_maximum_entries=2000,
                             xform_maximum_invocations_per_extraction=300):
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ResumeError("PDF защищён паролем. Загрузите незашифрованную копию.")
        if len(reader.pages) > 40:
            raise ResumeError("В PDF более 40 страниц. Загрузите короткую версию резюме.")
        pages, size = [], 0
        for page in reader.pages:
            chunk = page.extract_text() or ""
            size += len(chunk)
            if size > MAX_TEXT_CHARS:
                raise ResumeError("В PDF слишком много текста. Сократите резюме.")
            pages.append(chunk)
        text = "\n".join(pages)
        if len(text.strip()) < 20:
            raise ResumeError("В PDF нет извлекаемого текста: возможно, это скан. Вставьте текст резюме или загрузите DOCX/TXT.")
        return text


def validate_text(text):
    text = text.replace("\x00", "").strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ResumeError("Резюме слишком длинное. Оставьте не более 60 000 символов.")
    if len(text) < 20 or len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE)) < 12:
        raise ResumeError("В резюме слишком мало текста. Вставьте опыт работы и навыки (не менее 20 символов).")
    return text


def extract_resume(data, filename):
    if not data:
        raise ResumeError("Файл пуст. Выберите резюме с текстом.")
    if len(data) > MAX_FILE_BYTES:
        raise ResumeError("Размер файла превышает 8 МБ.")
    suffix = PurePath(filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise ResumeError("Поддерживаются PDF, DOCX и TXT. Сохраните резюме в одном из этих форматов.")
    try:
        if suffix == ".txt":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ResumeError("TXT должен быть в кодировке UTF-8. Пересохраните файл или вставьте текст.") from exc
        elif suffix == ".pdf":
            if not data.lstrip().startswith(b"%PDF-"):
                raise ResumeError("Не удалось распознать PDF. Проверьте файл или вставьте текст.")
            text = _pdf_text(data)
        else:
            with ZipFile(BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 3000 or sum(entry.file_size for entry in entries) > 30 * 1024 * 1024:
                    raise ResumeError("DOCX слишком большой после распаковки. Загрузите короткую версию.")
                item = archive.getinfo("word/document.xml")
                if item.file_size > 5 * 1024 * 1024:
                    raise ResumeError("В DOCX слишком много содержимого. Сократите резюме.")
                xml = archive.read(item)
                if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
                    raise ResumeError("Не удалось безопасно прочитать DOCX. Пересохраните документ.")
                root = ElementTree.fromstring(xml)
                namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = "\n".join("".join(node.itertext()) for node in root.findall(".//w:p", namespace))
        return validate_text(text)
    except ResumeError:
        raise
    except Exception as exc:
        raise ResumeError("Не удалось прочитать файл. Проверьте, что это корректный PDF/DOCX/TXT, или вставьте текст.") from exc
