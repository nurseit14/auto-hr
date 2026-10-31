import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowRight,
  BriefcaseBusiness,
  Check,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  LoaderCircle,
  MapPin,
  Search,
  Send,
  Upload,
} from "lucide-react";
import { api, json, safeSourceUrl, scoreLabel } from "./api";
import type { Candidate, Match, Pipeline, Vacancy, VacancyPage } from "./api";

const errorText = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Не удалось загрузить данные. Попробуйте ещё раз.";

function VacancyDetails({ vacancy }: { vacancy: Vacancy }) {
  const sourceUrl = safeSourceUrl(vacancy.url);
  return (
    <details className="vacancy-details">
      <summary>Требования и описание</summary>
      <div className="stack">
        {vacancy.requirements && (
          <div>
            <h4>Требования</h4>
            <p className="preserve-lines">{vacancy.requirements}</p>
          </div>
        )}
        {vacancy.responsibilities && (
          <div>
            <h4>Задачи</h4>
            <p className="preserve-lines">{vacancy.responsibilities}</p>
          </div>
        )}
        {vacancy.description && (
          <div>
            <h4>Описание</h4>
            <p className="preserve-lines">{vacancy.description}</p>
          </div>
        )}
        {sourceUrl && (
          <a
            href={sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            Открыть исходную публикацию <ExternalLink size={14} />
          </a>
        )}
      </div>
    </details>
  );
}

export function VacancyBrowser({
  onSelect,
  compact = false,
}: {
  onSelect: (vacancy: Vacancy) => void;
  compact?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [city, setCity] = useState("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<VacancyPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const limit = compact ? 8 : 12;
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const timer = window.setTimeout(
      () => {
        const params = new URLSearchParams({
          q: query,
          city,
          page: String(page),
          limit: String(limit),
        });
        api<VacancyPage>(`/vacancies?${params}`, { signal: controller.signal })
          .then((result) => {
            if (!controller.signal.aborted) setData(result);
          })
          .catch((reason) => {
            if (!controller.signal.aborted) setError(errorText(reason));
          })
          .finally(() => {
            if (!controller.signal.aborted) setLoading(false);
          });
      },
      query ? 300 : 0,
    );
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query, city, page, limit, retry]);

  const pages = Math.max(1, Math.ceil((data?.total || 0) / limit));
  return (
    <section className={`stack vacancy-browser${compact ? " compact" : ""}`}>
      <div className="notice">
        Архивные вакансии из датасета 2023 года используются для демонстрации.
        Свежесть публикаций проверяйте у источника.
      </div>
      <div className="toolbar">
        <label className="search-field">
          <Search size={18} />
          <span className="sr-only">Поиск вакансий</span>
          <input
            type="search"
            value={query}
            maxLength={200}
            placeholder="Должность, компания или технология"
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(1);
            }}
          />
        </label>
        <label className="field compact-field">
          <span className="sr-only">Город</span>
          <select
            value={city}
            onChange={(event) => {
              setCity(event.target.value);
              setPage(1);
            }}
          >
            <option value="">Все города</option>
            {data?.cities.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button
            className="button secondary"
            type="button"
            onClick={() => setRetry((value) => value + 1)}
          >
            Повторить
          </button>
        </div>
      )}
      <div className="catalogue-meta" role="status">
        {loading ? (
          <span>
            <LoaderCircle size={15} className="spin" /> Загружаю вакансии…
          </span>
        ) : (
          <span>Найдено: {data?.total.toLocaleString("ru-RU") || 0}</span>
        )}
      </div>
      <div className="result-list" aria-busy={loading}>
        {!loading && !error && data?.items.length === 0 && (
          <div className="empty">
            <Search size={30} />
            <h3>По этому запросу вакансий нет</h3>
            <p>Попробуйте другую технологию или выберите все города.</p>
            <button
              type="button"
              className="button secondary"
              onClick={() => {
                setQuery("");
                setCity("");
                setPage(1);
              }}
            >
              Сбросить фильтры
            </button>
          </div>
        )}
        {data?.items.map((vacancy) => (
          <article className="vacancy-row" key={vacancy.id}>
            <div className="vacancy-row-top">
              <div>
                <h3>{vacancy.title}</h3>
                <p className="muted">
                  {vacancy.company || "Компания не указана"}
                </p>
              </div>
              <span className={`tag${vacancy.is_custom ? " tag-success" : ""}`}>
                {vacancy.is_custom ? "Моя вакансия" : "Архив · 2023"}
              </span>
            </div>
            <div className="vacancy-meta">
              <span>
                <MapPin size={14} />
                {vacancy.city || "Город не указан"}
              </span>
              {vacancy.experience && (
                <span>
                  <BriefcaseBusiness size={14} />
                  {vacancy.experience}
                </span>
              )}
              {vacancy.salary && <strong>{vacancy.salary}</strong>}
            </div>
            {vacancy.skills.length > 0 && (
              <div className="tags">
                {vacancy.skills.slice(0, compact ? 6 : 10).map((skill) => (
                  <span className="tag" key={skill}>
                    {skill}
                  </span>
                ))}
                {vacancy.skills.length > (compact ? 6 : 10) && (
                  <span className="muted">
                    +{vacancy.skills.length - (compact ? 6 : 10)}
                  </span>
                )}
              </div>
            )}
            <VacancyDetails vacancy={vacancy} />
            <div className="vacancy-row-actions">
              <button
                className="button secondary"
                type="button"
                onClick={() => onSelect(vacancy)}
                disabled={loading}
              >
                Подобрать кандидатов <ArrowRight size={16} />
              </button>
            </div>
          </article>
        ))}
      </div>
      {data && data.total > limit && (
        <nav className="pagination" aria-label="Страницы вакансий">
          <button
            className="button secondary"
            disabled={loading || page <= 1}
            onClick={() => setPage((value) => value - 1)}
          >
            <ChevronLeft size={16} />
            Назад
          </button>
          <span>
            Страница {page} из {pages}
          </span>
          <button
            className="button secondary"
            disabled={loading || page >= pages}
            onClick={() => setPage((value) => value + 1)}
          >
            Далее
            <ChevronRight size={16} />
          </button>
        </nav>
      )}
    </section>
  );
}

export function MatchesView({
  candidates,
  onAddResume,
}: {
  candidates: Candidate[];
  onAddResume: () => void;
}) {
  const [candidateId, setCandidateId] = useState(candidates[0]?.id || "");
  const [city, setCity] = useState("");
  const [cities, setCities] = useState<string[]>([]);
  const [result, setResult] = useState<{
    items: Match[];
    pipeline: Pipeline;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  const selectedCandidate = candidates.find(
    (candidate) => candidate.id === candidateId,
  );
  const candidateSignature = selectedCandidate
    ? JSON.stringify([
        selectedCandidate.id,
        selectedCandidate.text,
        selectedCandidate.skills,
        selectedCandidate.clarifications,
      ])
    : "";
  useEffect(() => {
    const controller = new AbortController();
    api<VacancyPage>("/vacancies?limit=1", { signal: controller.signal })
      .then((data) => setCities(data.cities))
      .catch(() => {
        /* City is optional; the primary action reports server errors. */
      });
    return () => {
      controller.abort();
      request.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (!candidates.some((candidate) => candidate.id === candidateId)) {
      request.current?.abort();
      setLoading(false);
      setCandidateId(candidates[0]?.id || "");
      setResult(null);
    }
  }, [candidates, candidateId]);
  useEffect(() => {
    request.current?.abort();
    setLoading(false);
    setResult(null);
    setError("");
  }, [candidateSignature]);
  async function match(event: FormEvent) {
    event.preventDefault();
    if (!candidateId || loading) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await api<{ items: Match[]; pipeline: Pipeline }>(
        "/matches",
        {
          method: "POST",
          signal: controller.signal,
          body: json({
            candidate_id: candidateId,
            limit: 10,
            ...(city ? { city } : {}),
          }),
        },
      );
      if (!controller.signal.aborted) setResult(data);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorText(reason));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }
  return (
    <section className="stack matches-view">
      <p className="notice">
        В подборе есть архивные публикации 2023 года. Это проверка соответствия
        резюме, а не подтверждение открытой позиции.
      </p>
      {!candidates.length ? (
        <div className="empty">
          <BriefcaseBusiness size={36} />
          <h2>Начните с резюме</h2>
          <p>
            Добавьте опыт и IT-навыки, чтобы сравнить их с требованиями
            вакансий.
          </p>
          <button className="button primary" onClick={onAddResume}>
            <Upload size={17} />
            Добавить резюме
          </button>
        </div>
      ) : (
        <>
          <form
            className="match-form panel"
            onSubmit={match}
            aria-busy={loading}
          >
            <div className="form-grid">
              <label className="field">
                Резюме
                <select
                  value={candidateId}
                  disabled={loading}
                  onChange={(event) => {
                    setCandidateId(event.target.value);
                    setResult(null);
                    setError("");
                  }}
                >
                  {candidates.map((candidate) => (
                    <option key={candidate.id} value={candidate.id}>
                      {candidate.name}
                      {candidate.is_demo ? " · демо" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                Город
                <select
                  value={city}
                  disabled={loading}
                  onChange={(event) => {
                    setCity(event.target.value);
                    setResult(null);
                  }}
                >
                  <option value="">Все города</option>
                  {cities.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <button
              className="button primary"
              disabled={loading || !candidateId}
            >
              {loading ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <Search size={17} />
              )}
              {loading
                ? "Сравниваю с вакансиями…"
                : "Найти подходящие вакансии"}
            </button>
          </form>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          {loading && (
            <p className="muted" role="status">
              Извлекаю профессиональные признаки и ранжирую вакансии.
            </p>
          )}
          {result && (
            <>
              <div className="section-heading">
                <h2>Результаты подбора</h2>
                <span className="muted">{result.pipeline.label}</span>
              </div>
              <p className="muted">
                Балл ранжирования — относительное значение для сравнения
                вакансий, не вероятность трудоустройства.
              </p>
              {!result.items.length && (
                <div className="empty">
                  <Search size={30} />
                  <h3>Вакансий в этом городе не найдено</h3>
                  <p>Выберите все города и запустите подбор снова.</p>
                </div>
              )}
              <div className="result-list">
                {result.items.map((item, index) => (
                  <article
                    className="vacancy-row match-row"
                    key={item.vacancy.id}
                  >
                    <div className="vacancy-row-top">
                      <div className="ranked-title">
                        <span
                          className="rank-position"
                          aria-label={`Место ${index + 1}`}
                        >
                          {index + 1}
                        </span>
                        <div>
                          <h3>{item.vacancy.title}</h3>
                          <p className="muted">{item.vacancy.company}</p>
                        </div>
                      </div>
                      <div className="rank-value">
                        <strong>{scoreLabel(item.score)}</strong>
                        <span className="muted">
                          балл{" "}
                          {result.pipeline.ml2_available ? "ML2" : "подбора"}
                        </span>
                      </div>
                    </div>
                    <div className="vacancy-meta">
                      <span>
                        <MapPin size={14} />
                        {item.vacancy.city}
                      </span>
                      {item.vacancy.experience && (
                        <span>{item.vacancy.experience}</span>
                      )}
                      <span>
                        {item.vacancy.is_custom
                          ? "Вакансия вашего пространства"
                          : "Архив · 2023"}
                      </span>
                    </div>
                    <p>{item.explanation}</p>
                    <div className="match-skills">
                      <div>
                        <h4>Подтверждённые навыки</h4>
                        <div className="tags">
                          {item.matched_skills.length ? (
                            item.matched_skills.map((skill) => (
                              <span className="tag tag-success" key={skill}>
                                <Check size={12} />
                                {skill}
                              </span>
                            ))
                          ) : (
                            <span className="muted">
                              Совпадения не подтверждены
                            </span>
                          )}
                        </div>
                      </div>
                      <div>
                        <h4>Нужно уточнить</h4>
                        <div className="tags">
                          {item.missing_skills.length ? (
                            item.missing_skills.map((skill) => (
                              <span className="tag tag-warning" key={skill}>
                                {skill}
                              </span>
                            ))
                          ) : (
                            <span className="muted">
                              Все извлечённые требования покрыты
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <VacancyDetails vacancy={item.vacancy} />
                  </article>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}

export function FeedbackView({ onSaved }: { onSaved: () => void }) {
  const [rating, setRating] = useState<number | null>(null);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!rating) {
      setError("Выберите оценку от 1 до 5.");
      return;
    }
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await api<{ id: string }>("/feedback", {
        method: "POST",
        body: json({ rating, comment }),
      });
      setSaved(true);
      onSaved();
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="stack feedback-view">
      {saved ? (
        <div className="feedback-success panel" role="status">
          <Check size={30} />
          <h2>Отзыв сохранён</h2>
          <p>
            Спасибо. Ваша оценка и комментарий сохранены в локальной базе этого
            MVP.
          </p>
          <button
            className="button secondary"
            onClick={() => {
              setSaved(false);
              setRating(null);
              setComment("");
            }}
          >
            Оставить ещё один отзыв
          </button>
        </div>
      ) : (
        <form
          className="stack panel feedback-form"
          onSubmit={submit}
          aria-busy={busy}
        >
          <fieldset className="rating-fieldset" disabled={busy}>
            <legend>Насколько полезен результат подбора?</legend>
            <div className="rating-options">
              {[1, 2, 3, 4, 5].map((value) => (
                <label
                  key={value}
                  className={`rating-option${rating === value ? " selected" : ""}`}
                >
                  <input
                    type="radio"
                    name="rating"
                    value={value}
                    checked={rating === value}
                    onChange={() => setRating(value)}
                    required
                  />
                  <span>{value}</span>
                  <span className="sr-only">
                    {value === 1
                      ? " — совсем не полезен"
                      : value === 5
                        ? " — очень полезен"
                        : " из 5"}
                  </span>
                </label>
              ))}
            </div>
            <div className="rating-scale muted">
              <span>Совсем не полезен</span>
              <span>Очень полезен</span>
            </div>
          </fieldset>
          <label className="field">
            Что стоит улучшить? <span className="muted">необязательно</span>
            <textarea
              rows={6}
              maxLength={4000}
              value={comment}
              disabled={busy}
              onChange={(event) => setComment(event.target.value)}
              placeholder="Какие совпадения были полезны? Кого система недооценила? Какие вопросы помогли уточнить опыт?"
            />
          </label>
          <p className="muted">
            Не добавляйте контактные и другие личные данные кандидатов в
            комментарий.
          </p>
          {error && (
            <div className="form-error" role="alert">
              {error}
            </div>
          )}
          <div className="form-actions">
            <button className="button primary" disabled={busy}>
              {busy ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <Send size={17} />
              )}
              {busy ? "Сохраняю…" : "Отправить отзыв"}
            </button>
          </div>
        </form>
      )}
    </section>
  );
}
