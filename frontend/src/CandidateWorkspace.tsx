import { useEffect, useState } from "react";
import { ArrowRight, BookOpen, BriefcaseBusiness, ChevronRight, FileText, Info, LoaderCircle, LogOut, Menu, MessageSquare, Search, Trash2, Upload, X } from "lucide-react";
import { api } from "./api";
import type { AuthUser, Candidate, Vacancy } from "./api";
import { BrandMark } from "./LoginPage";
import { Modal, UploadDialog } from "./Dialogs";
import { FeedbackView, MatchesView, VacancyBrowser } from "./Views";
import { AdviceView } from "./AdviceView";

type View = "vacancies" | "matches" | "resumes" | "advice" | "rating";
const navigation = [
  { id: "vacancies" as View, label: "Вакансии", icon: BriefcaseBusiness },
  { id: "matches" as View, label: "Подбор вакансий", icon: BookOpen },
  { id: "resumes" as View, label: "Мои резюме", icon: FileText },
  { id: "advice" as View, label: "Разбор и вопросы", icon: MessageSquare },
];
const titles = { vacancies: "IT-вакансии", matches: "Подбор вакансий", resumes: "Мои резюме", advice: "Разбор резюме и вопросы", rating: "Оценить сервис" };
const subtitles = {
  vacancies: "Изучите требования и найдите роль для следующего шага.",
  matches: "Сравните ваш опыт с требованиями IT-вакансий.",
  resumes: "Ваш опыт, навыки и проекты в одном месте.",
  advice: "Проверьте основания подбора и уточните сведения о вакансии.",
  rating: "Расскажите, насколько полезен Auto-HR.",
};
const errorText = (error: unknown) => error instanceof Error ? error.message : "Не удалось обновить кабинет. Попробуйте ещё раз.";

export function CandidateWorkspace({ user, onLogout, loggingOut }: { user: AuthUser; onLogout: () => void; loggingOut: boolean }) {
  const [view, setView] = useState<View>("vacancies");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [revision, setRevision] = useState(0);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [upload, setUpload] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Candidate | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [query, setQuery] = useState("");
  const [adviceVacancy, setAdviceVacancy] = useState<Vacancy | null>(null);
  const [adviceCandidateId, setAdviceCandidateId] = useState("");
  const [matchesCandidateId, setMatchesCandidateId] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setReady(false);
    api<{ items: Candidate[] }>("/candidates", { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted) { setCandidates(result.items); setError(""); } })
      .catch(reason => { if (!controller.signal.aborted) setError(errorText(reason)); })
      .finally(() => { if (!controller.signal.aborted) setReady(true); });
    return () => controller.abort();
  }, [retry]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => { if (event.key === "Escape") setMobileOpen(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  async function refresh() {
    try { const result = await api<{ items: Candidate[] }>("/candidates"); setCandidates(result.items); }
    catch { setError("Изменения сохранены, но список не обновился. Обновите страницу."); }
    finally { setRevision(value => value + 1); }
  }
  function go(next: View) { setView(next); setMobileOpen(false); setError(""); }
  function openAdvice(vacancy: Vacancy | null, personId = "") {
    setAdviceVacancy(vacancy); setAdviceCandidateId(personId); go("advice");
  }
  async function remove() {
    if (!deleteTarget || deleting) return;
    setDeleting(true); setDeleteError("");
    try {
      await api(`/candidates/${encodeURIComponent(deleteTarget.id)}`, { method: "DELETE" });
      setCandidates(items => items.filter(item => item.id !== deleteTarget.id));
      setDeleteTarget(null); setNotice("Резюме и связанные с ним сведения удалены.");
      await refresh();
    } catch (reason) { setDeleteError(errorText(reason)); }
    finally { setDeleting(false); }
  }
  const filtered = candidates.filter(item => `${item.name} ${item.skills.join(" ")}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  return <div className="app-shell candidate-workspace">
    <a className="skip-link" href="#main">К содержимому</a>
    {mobileOpen && <button className="nav-scrim" aria-label="Закрыть меню" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar${mobileOpen ? " open" : ""}`}>
      <button className="brand" aria-label="Auto-HR — главная" onClick={() => go("vacancies")}><BrandMark /><span>Auto-HR</span></button>
      <nav aria-label="Основная навигация">{navigation.map(item => <button key={item.id} className={`nav-item${view === item.id ? " active" : ""}`} aria-current={view === item.id ? "page" : undefined} onClick={() => go(item.id)}><item.icon size={19} /><span>{item.label}</span>{item.id === "resumes" && candidates.length > 0 && <b>{candidates.length}</b>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="workspace-label"><span className="workspace-avatar">К</span><div><strong>{user.display_name}</strong><small>Кабинет кандидата</small></div></div><button className={`nav-item${view === "rating" ? " active" : ""}`} onClick={() => go("rating")}><MessageSquare size={19} />Оценить сервис</button><button className="nav-item" onClick={onLogout} disabled={loggingOut}>{loggingOut ? <LoaderCircle size={19} className="spin" /> : <LogOut size={19} />}Выйти</button><span className="sidebar-version">Auto-HR · IT-карьера</span></div>
    </aside>
    <div className="app-body">
      <header className="topbar"><div className="topbar-title"><button className="icon-button menu-button" aria-label="Открыть меню" onClick={() => setMobileOpen(true)}><Menu size={21} /></button><span>Кабинет кандидата</span><ChevronRight size={14} /><span className="muted">IT-карьера</span></div><div className="topbar-right"><span className="account-name">{user.username}</span><span className="user-avatar" aria-label="Кандидат">К</span><button className="icon-button account-logout" aria-label="Выйти из аккаунта" onClick={onLogout} disabled={loggingOut}><LogOut size={18} /></button></div></header>
      <main id="main" tabIndex={-1}>
        <div className="page-heading"><div><h1>{titles[view]}</h1><p>{subtitles[view]}</p></div>{view !== "rating" && <div className="heading-actions"><button className="button primary" onClick={() => setUpload(true)}><Upload size={17} />Загрузить резюме</button></div>}</div>
        {error && <div className="error-banner" role="alert"><Info size={18} /><span>{error}</span><button className="button secondary small" onClick={() => setRetry(value => value + 1)}>Повторить</button><button className="icon-button" aria-label="Скрыть ошибку" onClick={() => setError("")}><X size={17} /></button></div>}
        {notice && <div className="notice" role="status"><span>{notice}</span><button className="icon-button" aria-label="Скрыть сообщение" onClick={() => setNotice("")}><X size={17} /></button></div>}
        {!ready ? <div className="initial-loading"><LoaderCircle className="spin" size={26} /><p>Загружаем ваш кабинет…</p></div> : <>
          {view === "vacancies" && <VacancyBrowser selectLabel="Разбор и вопросы" customLabel="Вакансия HR" revision={revision} onSelect={item => openAdvice(item)} />}
          {view === "matches" && <MatchesView candidates={candidates} initialCandidateId={matchesCandidateId} customLabel="Вакансия HR" revision={revision} onAddResume={() => setUpload(true)} onAsk={(item, person) => openAdvice(item, person.id)} />}
          {view === "advice" && <AdviceView candidates={candidates} initialVacancy={adviceVacancy} initialCandidateId={adviceCandidateId} onAddResume={() => setUpload(true)} revision={revision} />}
          {view === "rating" && <FeedbackView onSaved={() => setNotice("Спасибо! Ваша оценка сохранена.")} />}
          {view === "resumes" && <section className="stack my-resumes">
            <div className="notice">Здесь находятся только резюме вашего кабинета. Подбор не отправляет их работодателю автоматически.</div>
            {candidates.length > 0 && <label className="search-field"><Search size={18} /><span className="sr-only">Поиск моих резюме</span><input type="search" placeholder="Имя или навык" value={query} onChange={event => setQuery(event.target.value)} /></label>}
            {!candidates.length && <div className="panel empty"><FileText size={34} /><h2>Добавьте ваше резюме</h2><p>Получите подбор вакансий, обратную связь и объяснение соответствия.</p><button className="button primary" onClick={() => setUpload(true)}><Upload size={17} />Загрузить резюме</button></div>}
            {filtered.map(item => <article className="panel my-resume" key={item.id}><div className="my-resume-heading"><div><h2>{item.name}</h2><p className="muted">{item.filename || "Резюме текстом"}</p></div><button className="button ghost danger-text" onClick={() => { setDeleteTarget(item); setDeleteError(""); }}><Trash2 size={16} />Удалить резюме</button></div><div className="tags">{item.skills.map(skill => <span className="tag" key={skill}>{skill}</span>)}</div><details className="my-resume-text"><summary>Посмотреть текст резюме</summary><div className="resume-text">{item.text}</div></details><div className="vacancy-row-actions"><button className="button primary" onClick={() => openAdvice(null, item.id)}>Получить разбор <ArrowRight size={16} /></button><button className="button secondary" onClick={() => { setMatchesCandidateId(item.id); go("matches"); }}>Подобрать вакансии</button></div></article>)}
            {candidates.length > 0 && !filtered.length && <div className="panel empty"><h2>Резюме не найдены</h2><p>Попробуйте другое имя или навык.</p><button className="button secondary" onClick={() => setQuery("")}>Сбросить поиск</button></div>}
          </section>}
        </>}
      </main>
    </div>
    {upload && <UploadDialog vacancies={[]} defaultVacancy={null} personal onClose={() => setUpload(false)} onCreated={items => { setCandidates(previous => [...items, ...previous.filter(item => !items.some(added => added.id === item.id))]); setNotice("Резюме сохранено в вашем кабинете."); refresh(); }} />}
    {deleteTarget && <Modal title="Удалить резюме?" onClose={() => { if (!deleting) setDeleteTarget(null); }}><p>Резюме «{deleteTarget.name}» и связанные с ним сведения будут удалены из вашего кабинета. Восстановить их нельзя.</p>{deleteError && <p className="form-error" role="alert">{deleteError}</p>}<div className="form-actions"><button className="button secondary" disabled={deleting} onClick={() => setDeleteTarget(null)}>Оставить резюме</button><button className="button danger" disabled={deleting} onClick={remove}>{deleting && <LoaderCircle size={17} className="spin" />}Удалить резюме</button></div></Modal>}
  </div>;
}
