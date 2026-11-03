import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { FileText, LoaderCircle, Search, Upload, X } from "lucide-react";
import { api, json } from "./api";
import type { Candidate, Vacancy, VacancyImportPreview } from "./api";

const errorText = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Не удалось сохранить данные. Попробуйте ещё раз.";

export function Modal({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const previous =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    return () => {
      element?.close();
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className={`modal${wide ? " modal-wide" : ""}`}
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="modal-heading">
        <h2 id={titleId}>{title}</h2>
        <button
          type="button"
          className="icon-button"
          onClick={onClose}
          aria-label="Закрыть окно"
        >
          <X size={20} />
        </button>
      </div>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}

export function UploadDialog({
  onClose,
  onCreated,
  vacancies,
  defaultVacancy,
  personal = false,
}: {
  onClose: () => void;
  onCreated: (items: Candidate[]) => void;
  vacancies: Vacancy[];
  defaultVacancy: Vacancy | null;
  personal?: boolean;
}) {
  const [vacancyId, setVacancyId] = useState(defaultVacancy?.id || "");
  const [mode, setMode] = useState<"files" | "text">("files");
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const close = () => {
    if (!busy) onClose();
  };

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setError("");
    if (mode === "text" && text.trim().length < 20) {
      setError(
        "Добавьте не меньше 20 символов: опыт, задачи и технические навыки.",
      );
      return;
    }
    if (mode === "files" && !files.length) {
      setError("Выберите хотя бы один файл резюме.");
      return;
    }
    setBusy(true);
    try {
      if (mode === "text") {
        const candidate = await api<Candidate>("/candidates", {
          method: "POST",
          body: json({
            name: name.trim() || "Кандидат",
            text: text.trim(),
            vacancy_id: vacancyId || null,
          }),
        });
        onCreated([candidate]);
        onClose();
      } else {
        const created: Candidate[] = [];
        const failed: File[] = [];
        const errors: string[] = [];
        for (const [index, file] of files.entries()) {
          setProgress(
            `Обрабатываю ${index + 1} из ${files.length}: ${file.name}`,
          );
          if (!/\.(pdf|docx|txt)$/i.test(file.name)) {
            failed.push(file);
            errors.push(`${file.name}: поддерживаются PDF, DOCX и TXT.`);
            continue;
          }
          if (file.size > 8 * 1024 * 1024) {
            failed.push(file);
            errors.push(`${file.name}: размер больше 8 МБ.`);
            continue;
          }
          try {
            const data = new FormData();
            data.append("file", file);
            if (vacancyId) data.append("vacancy_id", vacancyId);
            if (files.length === 1 && name.trim())
              data.append("name", name.trim());
            created.push(
              await api<Candidate>("/candidates/upload", {
                method: "POST",
                body: data,
              }),
            );
          } catch (reason) {
            failed.push(file);
            errors.push(`${file.name}: ${errorText(reason)}`);
          }
        }
        if (created.length) onCreated(created);
        if (!failed.length) onClose();
        else {
          setFiles(failed);
          if (fileInput.current) fileInput.current.value = "";
          setError(
            `${created.length ? `Добавлено резюме: ${created.length}. Остальные не добавлены.\n` : ""}${errors.join("\n")}`,
          );
        }
      }
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  return (
    <Modal title="Добавить резюме" onClose={close} wide>
      <form onSubmit={submit} className="stack" aria-busy={busy}>
        <p className="muted">
          {personal
            ? "Навыки будут извлечены автоматически. В разборе можно проверить результат и увидеть, какие сведения стоит уточнить в резюме."
            : "Загрузите опыт IT-специалиста. Навыки будут извлечены автоматически; результат можно проверить и исправить."}
        </p>
        {!personal && <label className="field">
          Куда добавить резюме
          <select
            value={vacancyId}
            onChange={(event) => setVacancyId(event.target.value)}
            disabled={busy}
          >
            <option value="">В общую базу · без вакансии</option>
            {vacancies.map((vacancy) => (
              <option value={vacancy.id} key={vacancy.id}>
                {vacancy.title} · {vacancy.company}
              </option>
            ))}
          </select>
          <span className="field-hint">
            {vacancyId
              ? "Кандидаты появятся в подборе только этой вакансии. В другие вакансии их можно добавить вручную."
              : "Резюме сохранится в разделе «Все кандидаты» и не попадёт в подбор автоматически."}
          </span>
        </label>}
        {personal && <p className="notice">Резюме сохранится в вашем кабинете. Сравнение с вакансией не отправляет его работодателю автоматически.</p>}
        <div className="segmented" aria-label="Способ добавления резюме">
          <button
            type="button"
            className={mode === "files" ? "active" : ""}
            aria-pressed={mode === "files"}
            disabled={busy}
            onClick={() => {
              setMode("files");
              setError("");
            }}
          >
            <Upload size={16} /> Загрузить файлы
          </button>
          <button
            type="button"
            className={mode === "text" ? "active" : ""}
            aria-pressed={mode === "text"}
            disabled={busy}
            onClick={() => {
              setMode("text");
              setError("");
            }}
          >
            <FileText size={16} /> Вставить текст
          </button>
        </div>
        {mode === "files" ? (
          <>
            <label className="upload-area">
              <Upload size={28} aria-hidden="true" />
              <strong>Выберите резюме с компьютера</strong>
              <span className="muted">PDF, DOCX или TXT · до 8 МБ на файл</span>
              <input
                ref={fileInput}
                type="file"
                accept=".pdf,.docx,.txt"
                multiple
                disabled={busy}
                onChange={(event) => {
                  setFiles(Array.from(event.target.files || []));
                  setError("");
                }}
              />
            </label>
            {files.length > 0 && (
              <ul className="file-list" aria-label="Выбранные файлы">
                {files.map((file, index) => (
                  <li key={`${file.name}-${index}`}>
                    <FileText size={16} />
                    <span>{file.name}</span>
                    <small className="muted">
                      {(file.size / 1024).toFixed(0)} КБ
                    </small>
                  </li>
                ))}
              </ul>
            )}
            <p className="muted">
              Для скана PDF нужен текстовый слой. Если текст не извлёкся,
              вставьте его вручную во второй вкладке.
            </p>
          </>
        ) : (
          <label className="field">
            Текст резюме
            <textarea
              required
              minLength={20}
              maxLength={60000}
              rows={11}
              value={text}
              onChange={(event) => setText(event.target.value)}
              disabled={busy}
              placeholder="Должность, опыт работы, проекты и технологии…"
            />
          </label>
        )}
        {(mode === "text" || files.length <= 1) && (
          <label className="field">
            Имя кандидата <span className="muted">необязательно</span>
            <input
              maxLength={120}
              value={name}
              disabled={busy}
              onChange={(event) => setName(event.target.value)}
              placeholder={
                mode === "files"
                  ? "По умолчанию — название файла"
                  : "Например, Александр К."
              }
            />
          </label>
        )}
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}
        {progress && (
          <p className="muted" role="status">
            {progress}
          </p>
        )}
        <div className="form-actions">
          <button
            type="button"
            className="button secondary"
            onClick={close}
            disabled={busy}
          >
            Отмена
          </button>
          <button className="button primary" disabled={busy}>
            {busy ? (
              <LoaderCircle size={17} className="spin" />
            ) : (
              <Upload size={17} />
            )}
            {busy
              ? "Обработка резюме…"
              : mode === "files" && files.length > 1
                ? `Добавить резюме (${files.length})`
                : "Добавить резюме"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function CandidatePoolDialog({
  vacancy,
  candidates,
  onClose,
  onSaved,
}: {
  vacancy: Vacancy;
  candidates: Candidate[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const available = candidates.filter(
    (candidate) => !candidate.vacancy_ids.includes(vacancy.id),
  );
  const visible = available.filter((candidate) =>
    `${candidate.name} ${candidate.skills.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || !selected.length) return;
    setBusy(true);
    setError("");
    const failed: string[] = [];
    let added = 0;
    let reason = "";
    for (const id of selected) {
      const candidate = available.find((item) => item.id === id);
      if (!candidate) continue;
      try {
        const latest = await api<Candidate>(
          `/candidates/${encodeURIComponent(id)}`,
        );
        await api<Candidate>(
          `/candidates/${encodeURIComponent(id)}/vacancies`,
          {
            method: "PUT",
            body: json({
              vacancy_ids: [...new Set([...latest.vacancy_ids, vacancy.id])],
            }),
          },
        );
        added += 1;
      } catch (error) {
        failed.push(id);
        reason = errorText(error);
      }
    }
    if (added) onSaved();
    if (failed.length) {
      setSelected(failed);
      setError(
        `${added ? `Добавлено: ${added}. ` : ""}Остальные кандидаты не добавлены. ${reason}`,
      );
    } else onClose();
    setBusy(false);
  }

  return (
    <Modal
      title="Добавить кандидатов из базы"
      onClose={() => {
        if (!busy) onClose();
      }}
      wide
    >
      <form className="stack" onSubmit={submit} aria-busy={busy}>
        <p className="muted">
          Вакансия: <strong>{vacancy.title}</strong>. Выберите резюме для этого
          подбора. Их связи с другими вакансиями сохранятся.
        </p>
        <label className="search-field">
          <Search size={17} />
          <input
            aria-label="Найти кандидата в базе"
            placeholder="Имя или навык"
            value={query}
            disabled={busy}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="pool-list">
          {visible.map((candidate) => (
            <label className="pool-option" key={candidate.id}>
              <input
                type="checkbox"
                checked={selected.includes(candidate.id)}
                disabled={busy}
                onChange={(event) =>
                  setSelected((old) =>
                    event.target.checked
                      ? [...old, candidate.id]
                      : old.filter((id) => id !== candidate.id),
                  )
                }
              />
              <span>
                <strong>{candidate.name}</strong>
                <small>
                  {candidate.skills.slice(0, 7).join(" · ") ||
                    "Навыки не извлечены"}
                </small>
                <small>
                  {candidate.vacancy_ids.length
                    ? `В других вакансиях: ${candidate.vacancy_ids.length}`
                    : "Без вакансии"}
                </small>
              </span>
            </label>
          ))}
          {!visible.length && (
            <div className="empty small-empty">
              <p>
                {available.length
                  ? "По этому запросу кандидатов нет."
                  : "Все резюме из базы уже добавлены в эту вакансию."}
              </p>
            </div>
          )}
        </div>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="form-actions">
          <button
            type="button"
            className="button secondary"
            disabled={busy}
            onClick={onClose}
          >
            Отмена
          </button>
          <button
            className="button primary"
            disabled={busy || !selected.length}
          >
            {busy && <LoaderCircle size={16} className="spin" />}Добавить
            выбранных{selected.length > 0 ? ` (${selected.length})` : ""}
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function VacancyDialog({
  onClose,
  onCreated,
  initialValues,
  importWarnings = [],
  filename,
}: {
  onClose: () => void;
  onCreated: (item: Vacancy) => void;
  initialValues?: VacancyImportPreview["fields"];
  importWarnings?: string[];
  filename?: string;
}) {
  const [values, setValues] = useState({
    title: "",
    company: "",
    city: "Алматы",
    requirements: "",
    description: "",
    experience: "",
    salary: "",
    ...initialValues,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const update = (key: keyof typeof values, value: string) =>
    setValues((previous) => ({ ...previous, [key]: value }));
  const close = () => {
    if (!busy) onClose();
  };
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const vacancy = await api<Vacancy>("/vacancies", {
        method: "POST",
        body: json(values),
      });
      onCreated(vacancy);
      onClose();
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal title={filename ? "Проверьте загруженную вакансию" : "Новая IT-вакансия"} onClose={close} wide>
      <form className="stack" onSubmit={submit} aria-busy={busy}>
        <p className="muted">
          Опишите роль и технические требования. По ним Auto-HR сравнит резюме и
          предложит уточняющие вопросы.
        </p>
        {filename && <p className="notice">Из файла «{filename}» извлечён текст. Проверьте поля и заполните недостающие сведения перед созданием вакансии.</p>}
        {importWarnings.length > 0 && <ul className="import-warnings">{importWarnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
        <label className="field">
          IT-должность
          <input
            required
            minLength={3}
            maxLength={180}
            value={values.title}
            disabled={busy}
            onChange={(event) => update("title", event.target.value)}
            placeholder="Например, Python Backend Developer"
          />
        </label>
        <div className="form-grid">
          <label className="field">
            Компания
            <input
              required
              maxLength={180}
              value={values.company}
              disabled={busy}
              onChange={(event) => update("company", event.target.value)}
              placeholder="Название компании"
            />
          </label>
          <label className="field">
            Город
            <input
              required
              maxLength={100}
              value={values.city}
              disabled={busy}
              onChange={(event) => update("city", event.target.value)}
              placeholder="Алматы, Астана или удалённо"
            />
          </label>
        </div>
        <label className="field">
          Технические требования
          <textarea
            required
            minLength={5}
            maxLength={12000}
            rows={5}
            value={values.requirements}
            disabled={busy}
            onChange={(event) => update("requirements", event.target.value)}
            placeholder="Обязательные технологии, практические задачи, требуемый опыт…"
          />
        </label>
        <label className="field">
          Задачи и описание <span className="muted">необязательно</span>
          <textarea
            maxLength={12000}
            rows={3}
            value={values.description}
            disabled={busy}
            onChange={(event) => update("description", event.target.value)}
            placeholder="Чем предстоит заниматься специалисту"
          />
        </label>
        <div className="form-grid">
          <label className="field">
            Опыт <span className="muted">необязательно</span>
            <input
              maxLength={180}
              value={values.experience}
              disabled={busy}
              onChange={(event) => update("experience", event.target.value)}
              placeholder="Например, от 1 до 3 лет"
            />
          </label>
          <label className="field">
            Зарплата <span className="muted">необязательно</span>
            <input
              maxLength={180}
              value={values.salary}
              disabled={busy}
              onChange={(event) => update("salary", event.target.value)}
              placeholder="Например, 500 000–800 000 ₸"
            />
          </label>
        </div>
        <p className="notice">
          После создания вакансия появится в каталоге кандидатов. Финальное решение по кандидатам принимает HR.
        </p>
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}
        <div className="form-actions">
          <button
            className="button secondary"
            type="button"
            disabled={busy}
            onClick={close}
          >
            Отмена
          </button>
          <button className="button primary" disabled={busy}>
            {busy && <LoaderCircle size={17} className="spin" />}
            {busy ? "Сохраняю…" : "Создать вакансию"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function CandidateEditDialog({
  candidate,
  onClose,
  onSaved,
}: {
  candidate: Candidate;
  onClose: () => void;
  onSaved: (item: Candidate) => void;
}) {
  const [name, setName] = useState(candidate.name);
  const [skills, setSkills] = useState(candidate.skills.join(", "));
  const [notes, setNotes] = useState(candidate.notes);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const close = () => {
    if (!busy) onClose();
  };
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    const values = Array.from(
      new Set(
        skills
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
      ),
    );
    if (
      values.length > 100 ||
      values.some((value) => value.length > 60 || value.includes("\n"))
    ) {
      setError(
        "Укажите до 100 навыков через запятую, каждый — не длиннее 60 символов.",
      );
      return;
    }
    setBusy(true);
    setError("");
    try {
      const item = await api<Candidate>(
        `/candidates/${encodeURIComponent(candidate.id)}`,
        {
          method: "PATCH",
          body: json({ name: name.trim(), skills: values, notes }),
        },
      );
      onSaved(item);
      onClose();
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal title="Проверка данных кандидата" onClose={close} wide>
      <form className="stack" onSubmit={submit} aria-busy={busy}>
        <label className="field">
          Имя кандидата
          <input
            required
            maxLength={120}
            value={name}
            disabled={busy}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label className="field">
          Навыки, подтверждённые HR
          <textarea
            rows={4}
            value={skills}
            disabled={busy}
            onChange={(event) => setSkills(event.target.value)}
            placeholder="Python, PostgreSQL, Docker"
          />
          <span className="muted">
            Перечислите навыки через запятую. Этот список заменит текущий список
            навыков кандидата.
          </span>
        </label>
        <p className="notice">
          Исходный текст резюме сохраняется. Добавленные вручную навыки
          отмечаются как подтверждённые HR; удалённые перестают учитываться как
          подтверждённые. Уточнения кандидата остаются отдельным источником.
        </p>
        <label className="field">
          Заметка HR
          <textarea
            maxLength={4000}
            rows={4}
            value={notes}
            disabled={busy}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Что проверили и что стоит обсудить на интервью"
          />
        </label>
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}
        <div className="form-actions">
          <button
            type="button"
            className="button secondary"
            disabled={busy}
            onClick={close}
          >
            Отмена
          </button>
          <button className="button primary" disabled={busy}>
            {busy && <LoaderCircle size={17} className="spin" />}
            {busy ? "Сохраняю…" : "Сохранить исправления"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
