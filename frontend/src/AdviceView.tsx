import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ArrowRight, BriefcaseBusiness, FileCheck2, LoaderCircle, MessageSquare, Send, Upload } from "lucide-react";
import { api, json, scoreLabel } from "./api";
import type { Candidate, ResumeFeedback, Vacancy, VacancyQuestion } from "./api";
import { Modal } from "./Dialogs";
import { VacancyBrowser } from "./Views";
import { ScoreBreakdown } from "./ScoreBreakdown";

const errorText = (reason: unknown) => reason instanceof Error ? reason.message : "Не удалось получить ответ. Попробуйте ещё раз.";
const sourceLabels = { vacancy: "Вакансия", resume: "Резюме", calculation: "Расчёт ML-2", clarification: "Ответ кандидата · самоотчёт", hr: "Подтверждение HR" };

export function AdviceView({ candidates, initialVacancy = null, initialCandidateId = "", onAddResume, revision = 0 }: {
  candidates: Candidate[];
  initialVacancy?: Vacancy | null;
  initialCandidateId?: string;
  onAddResume: () => void;
  revision?: number;
}) {
  const [vacancy, setVacancy] = useState<Vacancy | null>(initialVacancy);
  const [candidateId, setCandidateId] = useState(initialCandidateId || candidates[0]?.id || "");
  const [picker, setPicker] = useState(false);
  const [feedback, setFeedback] = useState<ResumeFeedback | null>(null);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackError, setFeedbackError] = useState("");
  const [questions, setQuestions] = useState<VacancyQuestion[]>([]);
  const [question, setQuestion] = useState("");
  const [questionBusy, setQuestionBusy] = useState(false);
  const [questionError, setQuestionError] = useState("");
  const [historyBusy, setHistoryBusy] = useState(false);
  const feedbackRequest = useRef<AbortController | null>(null);
  const questionRequest = useRef<AbortController | null>(null);
  const candidate = candidates.find(item => item.id === candidateId);
  const signature = candidate ? JSON.stringify([candidate.id, candidate.text, candidate.skills, candidate.clarifications]) : "";
  useEffect(() => { setVacancy(initialVacancy); }, [initialVacancy]);
  useEffect(() => { if (initialCandidateId && candidates.some(item => item.id === initialCandidateId)) setCandidateId(initialCandidateId); }, [initialCandidateId, candidates]);
  useEffect(() => {
    if (!candidates.some(item => item.id === candidateId)) setCandidateId(candidates[0]?.id || "");
  }, [candidateId, candidates]);
  useEffect(() => {
    feedbackRequest.current?.abort();
    questionRequest.current?.abort();
    setFeedback(null);
    setFeedbackError("");
    setQuestionError("");
    setFeedbackBusy(false);
    setQuestionBusy(false);
    setQuestions([]);
    if (!vacancy) { setHistoryBusy(false); return; }
    const controller = new AbortController();
    setHistoryBusy(true);
    const params = new URLSearchParams({ vacancy_id: vacancy.id });
    if (candidateId) params.set("candidate_id", candidateId);
    api<{ items: VacancyQuestion[] }>(`/questions?${params}`, { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted) setQuestions(previous => [...new Map([...result.items, ...previous].map(item => [item.id, item])).values()]); })
      .catch(reason => { if (!controller.signal.aborted) setQuestionError(errorText(reason)); })
      .finally(() => { if (!controller.signal.aborted) setHistoryBusy(false); });
    return () => controller.abort();
  }, [vacancy, candidateId, signature, revision]);
  useEffect(() => () => { feedbackRequest.current?.abort(); questionRequest.current?.abort(); }, []);
  async function analyze() {
    if (!candidate || feedbackBusy) return;
    feedbackRequest.current?.abort();
    const controller = new AbortController();
    feedbackRequest.current = controller;
    setFeedbackBusy(true);
    setFeedbackError("");
    setFeedback(null);
    try {
      const result = await api<ResumeFeedback>(`/candidates/${encodeURIComponent(candidate.id)}/feedback`, {
        method: "POST", signal: controller.signal, body: json(vacancy ? { vacancy_id: vacancy.id } : {}),
      });
      if (!controller.signal.aborted) setFeedback(result);
    } catch (reason) { if (!controller.signal.aborted) setFeedbackError(errorText(reason)); }
    finally { if (!controller.signal.aborted) setFeedbackBusy(false); }
  }
  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!vacancy || !question.trim() || questionBusy) return;
    const controller = new AbortController();
    questionRequest.current?.abort();
    questionRequest.current = controller;
    setQuestionBusy(true);
    setQuestionError("");
    const submitted = question.trim();
    try {
      const result = await api<VacancyQuestion>("/questions", {
        method: "POST", signal: controller.signal,
        body: json({ vacancy_id: vacancy.id, ...(candidate ? { candidate_id: candidate.id } : {}), question: submitted }),
      });
      if (!controller.signal.aborted) { setQuestions(previous => [...previous, result]); setQuestion(""); }
    } catch (reason) { if (!controller.signal.aborted) setQuestionError(errorText(reason)); }
    finally { if (!controller.signal.aborted) setQuestionBusy(false); }
  }
  const orderedQuestions = [...questions].sort((a, b) => a.created_at.localeCompare(b.created_at));
  return <section className="stack advice-view">
    <div className="panel advice-context">
      <div className="form-grid">
        <label className="field">Резюме для разбора
          <select value={candidateId} disabled={!candidates.length} onChange={event => setCandidateId(event.target.value)}>
            {!candidates.length && <option value="">Сначала добавьте резюме</option>}
            {candidates.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select>
        </label>
        <div className="field"><span>Вакансия</span><button type="button" className="advice-vacancy-choice" onClick={() => setPicker(true)}><span>{vacancy ? vacancy.title : "Выбрать вакансию для сравнения"}<small>{vacancy ? vacancy.company : "Можно получить разбор резюме и без вакансии"}</small></span><ArrowRight size={17} /></button></div>
      </div>
      {vacancy && <button className="text-button" onClick={() => setVacancy(null)}>Разобрать резюме без вакансии</button>}
    </div>
    <div className="advice-columns">
      <section className="panel advice-panel" aria-labelledby="resume-feedback-title">
        <div className="advice-panel-heading"><FileCheck2 size={21} /><h2 id="resume-feedback-title">Обратная связь по резюме</h2></div>
        <p className="muted">Проверьте, какие сведения уже есть в резюме и что стоит уточнить. С выбранной вакансией разбор включает соответствие её требованиям.</p>
        {!candidates.length ? <div className="empty small-empty"><p>Загрузите резюме, чтобы получить персональный разбор.</p><button className="button primary" onClick={onAddResume}><Upload size={17} /> Загрузить резюме</button></div> : <button className="button primary" disabled={feedbackBusy} onClick={analyze}>{feedbackBusy ? <LoaderCircle size={17} className="spin" /> : <FileCheck2 size={17} />}{feedbackBusy ? "Разбираем резюме…" : "Получить разбор резюме"}</button>}
        {feedbackError && <p className="form-error" role="alert">{feedbackError}</p>}
        {feedback && <div className="feedback-result" aria-live="polite">
          <p className="feedback-summary">{feedback.summary}</p>
          {feedback.source_note && <p className="muted">{feedback.source_note}</p>}
          {([['Что уже отражено', feedback.strengths], ['Что не подтверждено', feedback.gaps], ['Что можно улучшить', feedback.recommendations]] as const).map(([title, items]) => items.length > 0 && <section key={title}><h3>{title}</h3><ul>{items.map((item, index) => <li key={index}>{item}</li>)}</ul></section>)}
          {feedback.match && <><p className="advice-score">Балл по выбранной вакансии <strong>{scoreLabel(feedback.match.score)}</strong></p><ScoreBreakdown data={feedback.match.score_breakdown} /></>}
        </div>}
      </section>
      <section className="panel advice-panel" aria-labelledby="vacancy-questions-title">
        <div className="advice-panel-heading"><MessageSquare size={21} /><h2 id="vacancy-questions-title">Вопросы о вакансии и балле</h2></div>
        <p className="muted">Ответы строятся по тексту вакансии, выбранному резюме и расчёту подбора. Если сведений нет, помощник сообщит об этом.</p>
        {!vacancy ? <div className="empty small-empty"><BriefcaseBusiness size={28} /><p>Выберите вакансию, о которой хотите спросить.</p><button className="button secondary" onClick={() => setPicker(true)}>Выбрать вакансию <ArrowRight size={16} /></button></div> : <>
          <div className="question-suggestions" aria-label="Примеры вопросов">{["Какие навыки нужны для этой вакансии?", "Какая зарплата указана?", "Почему у меня такой балл?", "Как улучшить соответствие вакансии?"].map(value => <button key={value} className="question-suggestion" onClick={() => setQuestion(value)} disabled={questionBusy}>{value}</button>)}</div>
          {historyBusy && <p className="muted" role="status">Загружаем вопросы…</p>}
          <div className="question-history" aria-live="polite">{orderedQuestions.map(item => <article className="question-answer" key={item.id}>
            <h3>{item.question}</h3><p className="preserve-lines">{item.answer}</p>
            {item.evidence?.length > 0 && <details><summary>На чём основан ответ</summary>{item.evidence.map((proof, index) => <blockquote key={index}><p>{proof.quote}</p><cite>{sourceLabels[proof.source]}</cite></blockquote>)}</details>}
            {item.score_breakdown && <details><summary>Показать расчёт балла</summary><ScoreBreakdown data={item.score_breakdown} compact /></details>}
          </article>)}</div>
          <form onSubmit={ask} className="stack" aria-busy={questionBusy}>
            <label className="field">Ваш вопрос<textarea value={question} onChange={event => setQuestion(event.target.value)} required minLength={3} maxLength={1000} rows={3} disabled={questionBusy} placeholder="Например: какой опыт нужен и почему мой балл такой?" /></label>
            {questionError && <p className="form-error" role="alert">{questionError}</p>}
            <button className="button primary" disabled={questionBusy || question.trim().length < 3}>{questionBusy ? <LoaderCircle size={17} className="spin" /> : <Send size={17} />}{questionBusy ? "Готовим ответ…" : "Задать вопрос"}</button>
          </form>
        </>}
      </section>
    </div>
    {picker && <Modal title="Выберите IT-вакансию" onClose={() => setPicker(false)} wide><VacancyBrowser compact selectLabel="Выбрать вакансию" customLabel="Вакансия HR" revision={revision} onSelect={item => { setVacancy(item); setPicker(false); }} /></Modal>}
  </section>;
}
