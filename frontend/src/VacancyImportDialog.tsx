import { useState } from "react";
import type { FormEvent } from "react";
import { LoaderCircle, Upload } from "lucide-react";
import { api } from "./api";
import type { Vacancy, VacancyImportPreview } from "./api";
import { Modal, VacancyDialog } from "./Dialogs";

export function VacancyImportDialog({ onClose, onCreated }: { onClose: () => void; onCreated: (item: Vacancy) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<VacancyImportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || !file) return;
    setError("");
    if (!/\.(pdf|docx|txt)$/i.test(file.name)) { setError("Выберите PDF, DOCX или TXT с описанием IT-вакансии."); return; }
    if (file.size > 8 * 1024 * 1024) { setError("Файл больше 8 МБ. Выберите файл меньшего размера."); return; }
    setBusy(true);
    try {
      const data = new FormData(); data.append("file", file);
      const result = await api<VacancyImportPreview>("/vacancies/import-preview", { method: "POST", body: data });
      setPreview(result);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось прочитать файл. Попробуйте другой файл."); }
    finally { setBusy(false); }
  }
  if (preview) return <VacancyDialog initialValues={preview.fields} filename={preview.filename} importWarnings={preview.warnings} onClose={onClose} onCreated={onCreated} />;
  return <Modal title="Загрузить IT-вакансию" onClose={() => { if (!busy) onClose(); }}>
    <form className="stack" onSubmit={submit} aria-busy={busy}>
      <p className="muted">Выберите файл с описанием вакансии. Сначала откроется проверка полей — публикация произойдёт после создания вакансии.</p>
      <label className="upload-area"><Upload size={28} /><strong>Выберите файл вакансии</strong><span className="muted">PDF, DOCX или TXT · до 8 МБ</span><input type="file" accept=".pdf,.docx,.txt" disabled={busy} onChange={event => { setFile(event.target.files?.[0] || null); setError(""); }} /></label>
      {file && <p className="selected-import-file">{file.name}</p>}
      <p className="muted">PDF должен содержать текстовый слой. Если документ — скан, вставьте распознанный текст в форму новой вакансии.</p>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="form-actions"><button className="button secondary" type="button" disabled={busy} onClick={onClose}>Отмена</button><button className="button primary" disabled={busy || !file}>{busy ? <LoaderCircle size={17} className="spin" /> : <Upload size={17} />}{busy ? "Читаем вакансию…" : "Прочитать файл"}</button></div>
    </form>
  </Modal>;
}
