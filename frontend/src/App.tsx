import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BriefcaseBusiness,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Code2,
  Download,
  FileText,
  Info,
  Layers2,
  LoaderCircle,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  UserRound,
  UsersRound,
  X,
} from "lucide-react";
import { api, json, scoreLabel, statusLabel } from "./api";
import type {
  Candidate,
  Health,
  ReviewStatus,
  Shortlist,
  Stats,
  Vacancy,
  VacancyPage,
} from "./api";
import {
  CandidateEditDialog,
  Modal,
  UploadDialog,
  VacancyDialog,
} from "./Dialogs";
import { FeedbackView, MatchesView, VacancyBrowser } from "./Views";

type View = "shortlist" | "candidates" | "vacancies" | "matches" | "feedback";
type DetailTab = "evidence" | "resume" | "clarifications" | "notes";
const nav = [
  { id: "shortlist" as View, label: "Подбор кандидатов", icon: UsersRound },
  { id: "candidates" as View, label: "Все кандидаты", icon: FileText },
  {
    id: "vacancies" as View,
    label: "Каталог вакансий",
    icon: BriefcaseBusiness,
  },
  { id: "matches" as View, label: "Вакансии для кандидата", icon: BookOpen },
];
const titles: Record<View, string> = {
  shortlist: "Подбор кандидатов",
  candidates: "Все кандидаты",
  vacancies: "Каталог IT-вакансий",
  matches: "Вакансии для кандидата",
  feedback: "Обратная связь",
};
const subtitles: Record<View, string> = {
  shortlist: "Решение за вами. Основания — перед вами.",
  candidates: "Резюме, уточнения и ваши заметки в одном месте.",
  vacancies: "Ваши вакансии и архив для проверки подбора.",
  matches: "Найдите IT-роли, подходящие вашему опыту.",
  feedback: "Помогите проверить, насколько полезен Auto-HR.",
};
function initials(name: string) {
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((x) => x[0])
    .join("")
    .toLocaleUpperCase();
}
function message(error: unknown) {
  return error instanceof Error
    ? error.message
    : "Не удалось выполнить действие. Попробуйте ещё раз.";
}
function SkillTags({
  skills,
  tone = "neutral",
}: {
  skills: string[];
  tone?: string;
}) {
  return (
    <div className="tags">
      {skills.map((s) => (
        <span className={`tag ${tone}`} key={s}>
          {s}
        </span>
      ))}
    </div>
  );
}

function ResumeBody({ text, quote }: { text: string; quote: string }) {
  const index = quote ? text.indexOf(quote) : -1;
  return (
    <div className="resume-text">
      {index < 0 ? (
        text
      ) : (
        <>
          {text.slice(0, index)}
          <mark id="evidence-context" tabIndex={-1}>
            {quote}
          </mark>
          {text.slice(index + quote.length)}
        </>
      )}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>("shortlist");
  const [health, setHealth] = useState<Health | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [vacancy, setVacancy] = useState<Vacancy | null>(null);
  const [ranking, setRanking] = useState<Shortlist | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [revision, setRevision] = useState(0);
  const [rankingBusy, setRankingBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const [dialog, setDialog] = useState<
    "upload" | "vacancy" | "picker" | "edit" | "delete" | null
  >(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [tab, setTab] = useState<DetailTab>("evidence");
  const [resumeQuote, setResumeQuote] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileDetail, setMobileDetail] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [clarificationError, setClarificationError] = useState("");
  const [notes, setNotes] = useState("");
  const detailRef = useRef<HTMLElement>(null);
  useEffect(() => {
    setResumeQuote("");
  }, [selectedId]);
  useEffect(() => {
    if (tab === "resume" && resumeQuote) {
      const context = document.getElementById("evidence-context");
      context?.scrollIntoView({ block: "center" });
      context?.focus({ preventScroll: true });
    }
  }, [tab, resumeQuote]);
  const selectedCandidate = candidates.find((c) => c.id === selectedId) || null;
  const selectedRank =
    ranking?.candidates.find((r) => r.candidate.id === selectedId) || null;
  const isDemo = candidates.some((c) => c.is_demo);
  const hasDemoVacancy = Boolean(
    (vacancy as Vacancy & { is_demo?: boolean })?.is_demo,
  );

  async function reload() {
    const [c, s] = await Promise.all([
      api<{ items: Candidate[] }>("/candidates"),
      api<Stats>("/stats"),
    ]);
    setCandidates(c.items);
    setStats(s);
    return c.items;
  }
  function pickVacancy(v: Vacancy) {
    setVacancy(v);
    localStorage.setItem("autohr_vacancy", v.id);
    setRanking(null);
    setView("shortlist");
    setDialog(null);
    setMobileDetail(false);
  }
  function go(v: View) {
    setView(v);
    setMobileOpen(false);
    setMobileDetail(false);
    setError("");
  }
  async function updated(items?: Candidate[]) {
    if (items?.length) setSelectedId(items[items.length - 1].id);
    await reload();
    setRevision((v) => v + 1);
  }
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const h = await api<Health>("/health");
        if (!active) return;
        setHealth(h);
        const [c, s, j] = await Promise.all([
          api<{ items: Candidate[] }>("/candidates"),
          api<Stats>("/stats"),
          api<VacancyPage>("/vacancies?limit=100"),
        ]);
        if (!active) return;
        setCandidates(c.items);
        setStats(s);
        setSelectedId(c.items[0]?.id || "");
        const saved = localStorage.getItem("autohr_vacancy");
        let chosen =
          j.items.find((v) => v.id === saved) ||
          j.items.find((v) => v.is_custom) ||
          null;
        if (saved && !chosen) {
          try {
            chosen = await api<Vacancy>(
              `/vacancies/${encodeURIComponent(saved)}`,
            );
          } catch {
            localStorage.removeItem("autohr_vacancy");
          }
        }
        if (active) setVacancy(chosen);
      } catch (e) {
        if (active)
          setError("Не удалось подключиться к серверу. " + message(e));
      } finally {
        if (active) {
          setReady(true);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!ready || !vacancy) return;
    let active = true;
    setRankingBusy(true);
    api<Shortlist>("/shortlist", {
      method: "POST",
      body: json({ vacancy_id: vacancy.id }),
    })
      .then((data) => {
        if (!active) return;
        setRanking(data);
        setSelectedId((old) =>
          data.candidates.some((r) => r.candidate.id === old)
            ? old
            : data.candidates[0]?.candidate.id || "",
        );
      })
      .catch((e) => {
        if (active) setError(message(e));
      })
      .finally(() => {
        if (active) setRankingBusy(false);
      });
    return () => {
      active = false;
    };
  }, [ready, vacancy, revision]);
  useEffect(() => {
    setQuestion(
      selectedRank?.questions[0] ||
        "Расскажите о практическом опыте, важном для этой вакансии.",
    );
    setAnswer("");
    setClarificationError("");
    setNotes(selectedCandidate?.notes || "");
  }, [selectedId, selectedRank, selectedCandidate?.notes]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMobileOpen(false);
        setMobileDetail(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  async function demo() {
    setBusy("demo");
    setError("");
    try {
      const d = await api<{ vacancy_id: string; candidates: Candidate[] }>(
        "/demo",
        { method: "POST" },
      );
      await updated();
      pickVacancy(await api<Vacancy>(`/vacancies/${d.vacancy_id}`));
      setNotice(
        "Демо готово: вымышленные IT-кандидаты и тестовая вакансия. Можно пройти весь процесс подбора.",
      );
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy("");
    }
  }
  async function review(status: ReviewStatus) {
    if (!vacancy || !selectedCandidate) return;
    setBusy("review");
    try {
      await api("/reviews", {
        method: "PUT",
        body: json({
          vacancy_id: vacancy.id,
          candidate_id: selectedCandidate.id,
          status,
        }),
      });
      await updated();
      setNotice("Решение HR сохранено.");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy("");
    }
  }
  async function saveClarification(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedCandidate) return;
    setBusy("clarification");
    setClarificationError("");
    try {
      await api(`/candidates/${selectedCandidate.id}/clarifications`, {
        method: "POST",
        body: json({ question, answer }),
      });
      await updated();
      setAnswer("");
      setNotice("Ответ сохранён. Подбор пересчитан с учётом уточнения.");
    } catch (e) {
      setClarificationError(message(e));
    } finally {
      setBusy("");
    }
  }
  async function saveNotes(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedCandidate) return;
    setBusy("notes");
    try {
      await api(`/candidates/${selectedCandidate.id}`, {
        method: "PATCH",
        body: json({ notes }),
      });
      await updated();
      setNotice("Заметки сохранены.");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy("");
    }
  }
  async function remove() {
    if (!selectedCandidate) return;
    setBusy("delete");
    try {
      await api(`/candidates/${selectedCandidate.id}`, { method: "DELETE" });
      setDialog(null);
      setSelectedId("");
      setMobileDetail(false);
      await updated();
      setNotice("Резюме, уточнения и решения по кандидату удалены.");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy("");
    }
  }
  function chooseCandidate(id: string) {
    setSelectedId(id);
    setTab(view === "candidates" ? "resume" : "evidence");
    setMobileDetail(true);
    if (window.innerWidth < 900)
      setTimeout(
        () =>
          detailRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          }),
        30,
      );
  }
  const visibleRanks = (ranking?.candidates || []).filter(
    (r) =>
      (filter === "all" || r.status === filter) &&
      `${r.candidate.name} ${r.candidate.skills.join(" ")}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const visibleCandidates = candidates.filter((c) =>
    `${c.name} ${c.skills.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const noCandidates = candidates.length === 0;
  const showingWorkbench = view === "shortlist" || view === "candidates";
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        К содержимому
      </a>
      {mobileOpen && (
        <button
          className="nav-scrim"
          aria-label="Закрыть меню"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <button
          className="brand"
          onClick={() => go("shortlist")}
          aria-label="Auto-HR — главная"
        >
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>Auto-HR</span>
        </button>
        <nav aria-label="Основная навигация">
          {nav.map((n) => (
            <button
              key={n.id}
              className={`nav-item ${view === n.id ? "active" : ""}`}
              aria-current={view === n.id ? "page" : undefined}
              onClick={() => go(n.id)}
            >
              <n.icon size={19} />
              <span>{n.label}</span>
              {n.id === "candidates" && candidates.length > 0 && (
                <b>{candidates.length}</b>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="workspace-label">
            <span className="workspace-avatar">HR</span>
            <div>
              <strong>Рабочее пространство</strong>
              <small>IT-рекрутинг · Казахстан</small>
            </div>
          </div>
          <button
            className={`nav-item ${view === "feedback" ? "active" : ""}`}
            onClick={() => go("feedback")}
          >
            <MessageSquare size={19} />
            Обратная связь
          </button>
          <span className="sidebar-version">Auto-HR · MVP</span>
        </div>
      </aside>
      <div className="app-body">
        <header className="topbar">
          <div className="topbar-title">
            <button
              className="icon-button menu-button"
              aria-label="Открыть меню"
              onClick={() => setMobileOpen(true)}
            >
              <Menu size={21} />
            </button>
            <span>Кабинет HR</span>
            <ChevronRight size={14} />
            <span className="muted">
              {view === "matches" ? "Подбор вакансий" : "IT-рекрутинг"}
            </span>
          </div>
          <div className="topbar-right">
            <span className="local-status">
              <span />
              {health ? "Локальное пространство" : "Подключение"}
            </span>
            <span className="user-avatar" aria-label="HR">
              HR
            </span>
          </div>
        </header>
        <main id="main" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <h1>{titles[view]}</h1>
              <p>{subtitles[view]}</p>
            </div>
            <div className="heading-actions">
              {view !== "feedback" && (
                <>
                  <button
                    className="button primary"
                    onClick={() => setDialog("vacancy")}
                  >
                    <Plus size={17} />
                    Новая вакансия
                  </button>
                  <button
                    className="button secondary"
                    onClick={() => setDialog("upload")}
                  >
                    <Upload size={17} />
                    Загрузить резюме
                  </button>
                </>
              )}
            </div>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <Info size={18} />
              <span>{error}</span>
              <button
                className="icon-button"
                aria-label="Скрыть ошибку"
                onClick={() => setError("")}
              >
                <X size={17} />
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <Check size={18} />
              <span>{notice}</span>
              <button
                className="icon-button"
                aria-label="Скрыть сообщение"
                onClick={() => setNotice("")}
              >
                <X size={17} />
              </button>
            </div>
          )}
          {!ready ? (
            <div className="initial-loading">
              <LoaderCircle className="spin" size={26} />
              <p>Загружаем рабочее пространство…</p>
            </div>
          ) : (
            <>
              {showingWorkbench && (
                <>
                  {view === "shortlist" &&
                    (vacancy ? (
                      <section
                        className="vacancy-strip"
                        aria-label="Выбранная вакансия"
                      >
                        <span className="vacancy-symbol">
                          <Code2 size={23} />
                        </span>
                        <div className="vacancy-description">
                          <button
                            className="vacancy-title"
                            onClick={() => setDialog("picker")}
                          >
                            {vacancy.title}
                            <ChevronDown size={17} />
                          </button>
                          <div className="vacancy-meta">
                            {vacancy.company} <span>·</span> {vacancy.city}{" "}
                            <span>·</span>{" "}
                            {hasDemoVacancy
                              ? "Демо-вакансия"
                              : vacancy.is_custom
                                ? "Ваша вакансия"
                                : "Архив 2023"}
                          </div>
                          <SkillTags skills={vacancy.skills.slice(0, 8)} />
                        </div>
                        <button
                          className="button ghost small"
                          onClick={() => setDialog("picker")}
                        >
                          Сменить вакансию
                          <ArrowRight size={16} />
                        </button>
                      </section>
                    ) : (
                      <section className="choose-vacancy">
                        <span className="vacancy-symbol">
                          <BriefcaseBusiness size={24} />
                        </span>
                        <div>
                          <h2>С какой IT-вакансией работаем?</h2>
                          <p>
                            Создайте свою вакансию или выберите пример из
                            каталога.
                          </p>
                        </div>
                        <button
                          className="button secondary"
                          onClick={() => setDialog("picker")}
                        >
                          Выбрать вакансию
                          <ArrowRight size={17} />
                        </button>
                      </section>
                    ))}
                  {noCandidates ? (
                    <section className="intake-empty panel">
                      <div className="empty-visual">
                        <Layers2 size={38} strokeWidth={1.4} />
                        <span className="empty-check">
                          <Check size={15} />
                        </span>
                      </div>
                      <h2>Резюме — начало хорошего подбора</h2>
                      <p>
                        Добавьте IT-кандидатов. Auto-HR сопоставит навыки с
                        вакансией, покажет подтверждения и поможет уточнить
                        опыт.
                      </p>
                      <div className="empty-actions">
                        <button
                          className="button primary"
                          onClick={() => setDialog("upload")}
                        >
                          <Upload size={17} />
                          Добавить резюме
                        </button>
                        <button
                          className="button secondary"
                          onClick={demo}
                          disabled={Boolean(busy)}
                        >
                          {busy === "demo" ? (
                            <LoaderCircle className="spin" size={17} />
                          ) : (
                            <Sparkles size={17} />
                          )}
                          Посмотреть демо
                        </button>
                      </div>
                      <span className="upload-formats">
                        PDF, DOCX или TXT · до 8 МБ · можно вставить текст
                      </span>
                      <div className="intake-steps">
                        <span>
                          <BriefcaseBusiness size={17} />
                          Требования вакансии
                        </span>
                        <ChevronRight size={16} />
                        <span>
                          <FileText size={17} />
                          Подтверждения из резюме
                        </span>
                        <ChevronRight size={16} />
                        <span>
                          <CheckCheck size={17} />
                          Ваш шорт-лист
                        </span>
                      </div>
                    </section>
                  ) : view === "shortlist" && !vacancy ? (
                    <div className="panel empty">
                      <BriefcaseBusiness size={32} />
                      <h2>Резюме уже на месте</h2>
                      <p>
                        Выберите вакансию выше, чтобы сравнить{" "}
                        {candidates.length} кандидатов с её требованиями.
                      </p>
                      <button
                        className="button primary"
                        onClick={() => setDialog("picker")}
                      >
                        Выбрать вакансию
                      </button>
                    </div>
                  ) : (
                    <div
                      className={`workbench ${mobileDetail ? "show-detail" : ""}`}
                    >
                      <section
                        className="candidate-panel panel"
                        aria-label="Список кандидатов"
                      >
                        <div className="candidate-toolbar">
                          <h2>
                            Кандидаты <span>{candidates.length}</span>
                          </h2>
                          <div className="candidate-tools">
                            {view === "shortlist" && (
                              <>
                                <label className="compact-select">
                                  <SlidersHorizontal size={15} />
                                  <select
                                    aria-label="Фильтр по статусу"
                                    value={filter}
                                    onChange={(e) => setFilter(e.target.value)}
                                  >
                                    <option value="all">Все статусы</option>
                                    <option value="new">На рассмотрении</option>
                                    <option value="shortlisted">
                                      В шорт-листе
                                    </option>
                                    <option value="hold">
                                      Вернуться позже
                                    </option>
                                  </select>
                                </label>
                                <a
                                  className="button secondary small"
                                  href={`/api/export?vacancy_id=${encodeURIComponent(vacancy!.id)}`}
                                  download
                                >
                                  <Download size={15} />
                                  CSV
                                </a>
                              </>
                            )}
                          </div>
                        </div>
                        <label className="candidate-search">
                          <Search size={16} />
                          <input
                            aria-label="Поиск кандидатов"
                            placeholder="Имя или навык"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                          />
                          {query && (
                            <button
                              className="icon-button"
                              aria-label="Очистить поиск"
                              onClick={() => setQuery("")}
                            >
                              <X size={15} />
                            </button>
                          )}
                        </label>
                        {view === "shortlist" && (
                          <div className="list-caption">
                            <span>
                              {rankingBusy
                                ? "Пересчитываем подбор…"
                                : "По релевантности"}
                            </span>
                            <span title="Относительный балл модели, не вероятность найма">
                              Балл ML2 <CircleHelp size={12} />
                            </span>
                          </div>
                        )}
                        {rankingBusy && !ranking ? (
                          <div className="empty">
                            <LoaderCircle className="spin" size={24} />
                            <p>Сравниваем профессиональные признаки…</p>
                          </div>
                        ) : view === "shortlist" ? (
                          <div className="candidate-list">
                            {visibleRanks.map((r, i) => (
                              <button
                                key={r.candidate.id}
                                className={`candidate-row ${selectedId === r.candidate.id ? "selected" : ""}`}
                                onClick={() => chooseCandidate(r.candidate.id)}
                                aria-pressed={selectedId === r.candidate.id}
                              >
                                <span
                                  className={`candidate-avatar color-${i % 4}`}
                                >
                                  {initials(r.candidate.name)}
                                </span>
                                <span className="candidate-row-main">
                                  <strong>
                                    {r.candidate.name}
                                    {r.candidate.is_demo && (
                                      <small className="demo-label">Демо</small>
                                    )}
                                  </strong>
                                  <span className="candidate-file">
                                    {r.candidate.filename || "Резюме текстом"}
                                  </span>
                                  <span className={`review-badge ${r.status}`}>
                                    {statusLabel[r.status]}
                                  </span>
                                </span>
                                <span className="candidate-coverage">
                                  <span>
                                    <b>
                                      {r.matched_skills.length} из{" "}
                                      {vacancy?.skills.length || 0}
                                    </b>{" "}
                                    навыков
                                  </span>
                                  <span className="tags">
                                    {r.matched_skills.slice(0, 4).map((s) => (
                                      <span className="tag confirmed" key={s}>
                                        {s}
                                      </span>
                                    ))}
                                    {r.uncertain_skills.slice(0, 1).map((s) => (
                                      <span className="tag uncertain" key={s}>
                                        {s}
                                      </span>
                                    ))}
                                  </span>
                                </span>
                                <span className="candidate-score">
                                  {scoreLabel(r.score)}
                                  <ChevronRight size={16} />
                                </span>
                              </button>
                            ))}
                            {!visibleRanks.length && (
                              <div className="empty small-empty">
                                <Search size={24} />
                                <h3>Кандидаты не найдены</h3>
                                <p>Измените поиск или выберите все статусы.</p>
                                <button
                                  className="button ghost"
                                  onClick={() => {
                                    setFilter("all");
                                    setQuery("");
                                  }}
                                >
                                  Сбросить фильтры
                                </button>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="candidate-list">
                            {visibleCandidates.map((c, i) => (
                              <button
                                key={c.id}
                                className={`candidate-row all-candidates ${selectedId === c.id ? "selected" : ""}`}
                                onClick={() => chooseCandidate(c.id)}
                                aria-pressed={selectedId === c.id}
                              >
                                <span
                                  className={`candidate-avatar color-${i % 4}`}
                                >
                                  {initials(c.name)}
                                </span>
                                <span className="candidate-row-main">
                                  <strong>
                                    {c.name}
                                    {c.is_demo && (
                                      <small className="demo-label">Демо</small>
                                    )}
                                  </strong>
                                  <span className="candidate-file">
                                    {c.filename || "Резюме текстом"}
                                  </span>
                                  <SkillTags skills={c.skills.slice(0, 5)} />
                                </span>
                                <ChevronRight size={17} />
                              </button>
                            ))}
                            {!visibleCandidates.length && (
                              <div className="empty">
                                <p>Измените имя или навык в поиске.</p>
                              </div>
                            )}
                          </div>
                        )}
                        <div className="candidate-panel-footer">
                          <Info size={15} />
                          <span>
                            {view === "shortlist"
                              ? "Все кандидаты доступны для просмотра. Балл помогает сравнивать и не заменяет решение HR."
                              : "Резюме сохраняются в вашем локальном пространстве."}
                          </span>
                        </div>
                      </section>
                      <section
                        className="detail-panel panel"
                        ref={detailRef}
                        aria-label="Разбор кандидата"
                      >
                        {selectedCandidate ? (
                          <>
                            <button
                              className="button ghost mobile-back"
                              onClick={() => setMobileDetail(false)}
                            >
                              <ArrowLeft size={16} />К списку кандидатов
                            </button>
                            <div className="detail-header">
                              <span className="candidate-avatar color-0">
                                {initials(selectedCandidate.name)}
                              </span>
                              <div>
                                <h2>{selectedCandidate.name}</h2>
                                <p>
                                  {selectedCandidate.is_demo
                                    ? "Демонстрационное резюме"
                                    : selectedCandidate.filename ||
                                      "Резюме текстом"}
                                </p>
                              </div>
                              <button
                                className="icon-button"
                                aria-label="Изменить данные кандидата"
                                title="Исправить навыки и данные"
                                onClick={() => setDialog("edit")}
                              >
                                <Pencil size={17} />
                              </button>
                            </div>
                            <div
                              className="detail-tabs"
                              role="tablist"
                              aria-label="Данные кандидата"
                            >
                              {(
                                [
                                  { id: "evidence", label: "Разбор" },
                                  { id: "resume", label: "Резюме" },
                                  { id: "clarifications", label: "Уточнения" },
                                  { id: "notes", label: "Заметки" },
                                ] as const
                              ).map((t, i, items) => (
                                <button
                                  id={`tab-${t.id}`}
                                  key={t.id}
                                  role="tab"
                                  aria-selected={tab === t.id}
                                  tabIndex={tab === t.id ? 0 : -1}
                                  aria-controls="candidate-tabpanel"
                                  className={tab === t.id ? "active" : ""}
                                  onClick={() => setTab(t.id)}
                                  onKeyDown={(e) => {
                                    if (
                                      e.key === "ArrowRight" ||
                                      e.key === "ArrowLeft"
                                    ) {
                                      e.preventDefault();
                                      const next =
                                        items[
                                          (i +
                                            (e.key === "ArrowRight"
                                              ? 1
                                              : items.length - 1)) %
                                            items.length
                                        ];
                                      setTab(next.id);
                                      document
                                        .getElementById(`tab-${next.id}`)
                                        ?.focus();
                                    }
                                  }}
                                >
                                  {t.label}
                                  {t.id === "clarifications" &&
                                    selectedCandidate.clarifications.length >
                                      0 && (
                                      <span>
                                        {
                                          selectedCandidate.clarifications
                                            .length
                                        }
                                      </span>
                                    )}
                                </button>
                              ))}
                            </div>
                            <div
                              className="detail-content"
                              id="candidate-tabpanel"
                              role="tabpanel"
                              aria-labelledby={`tab-${tab}`}
                              key={`${selectedCandidate.id}-${tab}`}
                            >
                              {tab === "evidence" &&
                                (selectedRank && view === "shortlist" ? (
                                  <>
                                    <div className="coverage-summary">
                                      <span>
                                        <strong>
                                          {selectedRank.matched_skills.length}
                                        </strong>{" "}
                                        из {vacancy?.skills.length || 0} навыков
                                        подтверждено
                                      </span>
                                      <span
                                        className="model-score"
                                        title="Относительная оценка ранжирования, не вероятность"
                                      >
                                        ML2 {scoreLabel(selectedRank.score)}
                                      </span>
                                    </div>
                                    <p className="explanation">
                                      {selectedRank.explanation}
                                    </p>
                                    <div className="evidence-section">
                                      <h3>
                                        Подтверждено{" "}
                                        <span>
                                          {selectedRank.matched_skills.length}
                                        </span>
                                      </h3>
                                      <p className="section-hint">
                                        Основания из профессиональной части
                                        резюме
                                      </p>
                                      {selectedRank.evidence
                                        .filter((e) =>
                                          selectedRank.matched_skills.includes(
                                            e.skill,
                                          ),
                                        )
                                        .map((e, i) => (
                                          <div
                                            className="evidence-item"
                                            key={`${e.skill}-${i}`}
                                          >
                                            <Check
                                              className="evidence-icon confirmed"
                                              size={19}
                                            />
                                            <div>
                                              <div className="evidence-title">
                                                <strong>{e.skill}</strong>
                                                {e.source === "resume" && (
                                                  <button
                                                    className="source-context"
                                                    onClick={() => {
                                                      setResumeQuote(e.quote);
                                                      setTab("resume");
                                                    }}
                                                  >
                                                    Показать в резюме
                                                    <ArrowRight size={12} />
                                                  </button>
                                                )}
                                              </div>
                                              <blockquote>{e.quote}</blockquote>
                                              <span className="evidence-source">
                                                {e.source === "clarification"
                                                  ? "Ответ кандидата · самоотчёт"
                                                  : e.source === "hr"
                                                    ? "Подтверждено HR"
                                                    : "Резюме"}
                                              </span>
                                            </div>
                                          </div>
                                        ))}
                                      {!selectedRank.matched_skills.length && (
                                        <p className="section-hint">
                                          Явных подтверждений пока нет.
                                          Проверьте резюме и уточните опыт.
                                        </p>
                                      )}
                                    </div>
                                    {(selectedRank.missing_skills.length > 0 ||
                                      selectedRank.uncertain_skills.length >
                                        0) && (
                                      <div className="evidence-section">
                                        <h3>
                                          Нужно уточнить{" "}
                                          <span>
                                            {selectedRank.missing_skills
                                              .length +
                                              selectedRank.uncertain_skills
                                                .length}
                                          </span>
                                        </h3>
                                        <p className="section-hint">
                                          Недостаточно данных — это не отказ
                                        </p>
                                        <SkillTags
                                          skills={Array.from(
                                            new Set([
                                              ...selectedRank.uncertain_skills,
                                              ...selectedRank.missing_skills,
                                            ]),
                                          )}
                                          tone="uncertain"
                                        />
                                        <button
                                          className="button secondary small clarify-link"
                                          onClick={() =>
                                            setTab("clarifications")
                                          }
                                        >
                                          <MessageSquare size={15} />
                                          Уточнить опыт
                                          <ArrowRight size={15} />
                                        </button>
                                      </div>
                                    )}
                                    <button
                                      className="text-button"
                                      onClick={() => setDialog("edit")}
                                    >
                                      <Pencil size={14} />
                                      Исправить распознанные навыки
                                    </button>
                                  </>
                                ) : (
                                  <div className="empty small-empty">
                                    <BriefcaseBusiness size={28} />
                                    <h3>Для разбора нужна вакансия</h3>
                                    <p>
                                      Выберите IT-вакансию, чтобы увидеть
                                      подтверждения по её требованиям.
                                    </p>
                                    <button
                                      className="button secondary"
                                      onClick={() => setDialog("picker")}
                                    >
                                      Выбрать вакансию
                                    </button>
                                  </div>
                                ))}
                              {tab === "resume" && (
                                <>
                                  <div className="section-toolbar">
                                    <h3>Профессиональный профиль</h3>
                                    <button
                                      className="button ghost small"
                                      onClick={() => setDialog("edit")}
                                    >
                                      <Pencil size={14} />
                                      Исправить
                                    </button>
                                  </div>
                                  <SkillTags
                                    skills={selectedCandidate.skills}
                                  />
                                  <ResumeBody
                                    text={selectedCandidate.text}
                                    quote={resumeQuote}
                                  />
                                  <p className="section-hint">
                                    Распознанные контакты скрыты автоматически.
                                    Проверьте текст, если он содержит личные
                                    сведения.
                                  </p>
                                </>
                              )}
                              {tab === "clarifications" && (
                                <>
                                  <h3>Уточнить практический опыт</h3>
                                  <p className="section-hint">
                                    Передайте вопрос кандидату и сохраните его
                                    ответ. HR сможет проверить самоотчёт при
                                    интервью.
                                  </p>
                                  {selectedRank?.questions &&
                                    selectedRank.questions.length > 1 && (
                                      <label className="field">
                                        Вопрос по вакансии
                                        <select
                                          value={question}
                                          onChange={(e) =>
                                            setQuestion(e.target.value)
                                          }
                                        >
                                          {selectedRank.questions.map((q) => (
                                            <option key={q}>{q}</option>
                                          ))}
                                        </select>
                                      </label>
                                    )}
                                  <form
                                    className="stack"
                                    onSubmit={saveClarification}
                                  >
                                    <label className="field">
                                      Вопрос кандидату
                                      <textarea
                                        value={question}
                                        onChange={(e) =>
                                          setQuestion(e.target.value)
                                        }
                                        minLength={5}
                                        maxLength={1000}
                                        required
                                        rows={3}
                                      />
                                    </label>
                                    <label className="field">
                                      Ответ кандидата
                                      <textarea
                                        value={answer}
                                        onChange={(e) =>
                                          setAnswer(e.target.value)
                                        }
                                        placeholder="Какая задача, какие технологии, личный вклад и результат?"
                                        minLength={3}
                                        maxLength={5000}
                                        required
                                        rows={4}
                                      />
                                    </label>
                                    {clarificationError && (
                                      <p className="form-error" role="alert">
                                        {clarificationError}
                                      </p>
                                    )}
                                    <button
                                      className="button primary"
                                      disabled={Boolean(busy)}
                                    >
                                      {busy === "clarification" ? (
                                        <LoaderCircle
                                          className="spin"
                                          size={17}
                                        />
                                      ) : (
                                        <Check size={17} />
                                      )}
                                      Сохранить ответ и пересчитать
                                    </button>
                                  </form>
                                  {selectedCandidate.clarifications.length >
                                    0 && (
                                    <div className="clarification-history">
                                      <h3>Сохранённые уточнения</h3>
                                      {selectedCandidate.clarifications.map(
                                        (c, i) => (
                                          <article key={i}>
                                            <strong>{c.question}</strong>
                                            <p>{c.answer}</p>
                                            <span className="evidence-source">
                                              Со слов кандидата
                                            </span>
                                          </article>
                                        ),
                                      )}
                                    </div>
                                  )}
                                </>
                              )}
                              {tab === "notes" && (
                                <form className="stack" onSubmit={saveNotes}>
                                  <h3>Заметки HR</h3>
                                  <p className="section-hint">
                                    Сохраняйте профессиональные наблюдения и
                                    вопросы для интервью. Заметки не влияют на
                                    ранжирование.
                                  </p>
                                  <label className="field">
                                    Комментарий к кандидату
                                    <textarea
                                      value={notes}
                                      onChange={(e) => setNotes(e.target.value)}
                                      placeholder="Что обсудить на интервью?"
                                      maxLength={4000}
                                      rows={7}
                                    />
                                  </label>
                                  <button
                                    className="button primary"
                                    disabled={Boolean(busy)}
                                  >
                                    {busy === "notes" ? (
                                      <LoaderCircle
                                        className="spin"
                                        size={17}
                                      />
                                    ) : (
                                      <Check size={17} />
                                    )}
                                    Сохранить заметки
                                  </button>
                                  <button
                                    type="button"
                                    className="text-button danger-text"
                                    onClick={() => setDialog("delete")}
                                  >
                                    <Trash2 size={15} />
                                    Удалить резюме и связанные данные
                                  </button>
                                </form>
                              )}
                            </div>
                            {view === "shortlist" && selectedRank && (
                              <div className="detail-decision">
                                <label
                                  className="sr-only"
                                  htmlFor="review-status"
                                >
                                  Решение HR
                                </label>
                                <select
                                  id="review-status"
                                  value={selectedRank.status}
                                  onChange={(e) =>
                                    review(e.target.value as ReviewStatus)
                                  }
                                  disabled={Boolean(busy)}
                                >
                                  {Object.entries(statusLabel).map(([k, v]) => (
                                    <option key={k} value={k}>
                                      {v}
                                    </option>
                                  ))}
                                </select>
                                <button
                                  className="button secondary"
                                  onClick={() => setTab("clarifications")}
                                >
                                  <MessageSquare size={16} />
                                  Уточнить опыт
                                </button>
                                <button
                                  className="button primary"
                                  disabled={
                                    Boolean(busy) ||
                                    selectedRank.status === "shortlisted"
                                  }
                                  onClick={() => review("shortlisted")}
                                >
                                  {busy === "review" ? (
                                    <LoaderCircle className="spin" size={16} />
                                  ) : (
                                    <CheckCheck size={16} />
                                  )}
                                  {selectedRank.status === "shortlisted"
                                    ? "В шорт-листе"
                                    : "В шорт-лист"}
                                </button>
                              </div>
                            )}
                          </>
                        ) : (
                          <div className="empty">
                            <UserRound size={29} />
                            <h3>Выберите кандидата</h3>
                            <p>Откроем его резюме и основания подбора.</p>
                          </div>
                        )}
                      </section>
                    </div>
                  )}
                  {isDemo && (
                    <div className="demo-line">
                      <Sparkles size={14} />
                      <span>
                        В пространстве есть вымышленные демо-кандидаты. Их можно
                        удалить или добавить свои резюме.
                      </span>
                      <button onClick={demo} disabled={Boolean(busy)}>
                        Открыть демо
                      </button>
                    </div>
                  )}
                  <div className="workspace-footer">
                    <span>
                      <ShieldCheck size={15} />
                      Финальное решение всегда принимает HR
                    </span>
                    {health && (
                      <span>
                        {health.ml2_available
                          ? "ML2 подключена"
                          : "Модель недоступна"}{" "}
                        ·{" "}
                        {health.ml1_available
                          ? "семантический подбор"
                          : "подбор по тексту и навыкам"}
                      </span>
                    )}
                    {stats && (
                      <span>В шорт-листах: {stats.shortlisted_count}</span>
                    )}
                  </div>
                </>
              )}
              {view === "vacancies" && (
                <VacancyBrowser onSelect={pickVacancy} />
              )}
              {view === "matches" && (
                <MatchesView
                  candidates={candidates}
                  onAddResume={() => setDialog("upload")}
                />
              )}
              {view === "feedback" && (
                <FeedbackView
                  onSaved={() => {
                    reload();
                  }}
                />
              )}
            </>
          )}
        </main>
      </div>
      {dialog === "upload" && (
        <UploadDialog
          onClose={() => setDialog(null)}
          onCreated={(items) => {
            updated(items);
            setNotice("Резюме добавлены в рабочее пространство.");
          }}
        />
      )}
      {dialog === "vacancy" && (
        <VacancyDialog
          onClose={() => setDialog(null)}
          onCreated={(v) => {
            pickVacancy(v);
            setRevision((x) => x + 1);
            reload();
            setNotice(
              "IT-вакансия создана. Теперь добавьте кандидатов или сравните уже сохранённые резюме.",
            );
          }}
        />
      )}
      {dialog === "picker" && (
        <Modal
          title="Выберите IT-вакансию"
          onClose={() => setDialog(null)}
          wide
        >
          <VacancyBrowser onSelect={pickVacancy} compact />
        </Modal>
      )}
      {dialog === "edit" && selectedCandidate && (
        <CandidateEditDialog
          candidate={selectedCandidate}
          onClose={() => setDialog(null)}
          onSaved={() => {
            updated();
            setNotice("Данные кандидата исправлены. Подбор пересчитывается.");
          }}
        />
      )}
      {dialog === "delete" && selectedCandidate && (
        <Modal title="Удалить резюме?" onClose={() => setDialog(null)}>
          <p>
            Резюме «{selectedCandidate.name}», уточнения, заметки и решения по
            нему будут удалены из рабочего пространства.
          </p>
          <div className="form-actions">
            <button
              className="button secondary"
              onClick={() => setDialog(null)}
            >
              Оставить резюме
            </button>
            <button
              className="button danger"
              onClick={remove}
              disabled={Boolean(busy)}
            >
              <Trash2 size={16} />
              Удалить данные
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
