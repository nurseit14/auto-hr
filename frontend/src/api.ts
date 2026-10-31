export interface Vacancy {
  id: string;
  title: string;
  company: string;
  city: string;
  salary: string;
  experience: string;
  employment: string;
  description: string;
  requirements: string;
  responsibilities: string;
  skills: string[];
  url: string;
  is_custom: boolean;
  publish_date: string;
}
export interface Candidate {
  id: string;
  name: string;
  filename: string;
  text: string;
  skills: string[];
  notes: string;
  is_demo: boolean;
  created_at: string;
  clarifications: { question: string; answer: string }[];
}
export type ReviewStatus = "new" | "shortlisted" | "hold";
export interface RankedCandidate {
  candidate: Candidate;
  score: number;
  skill_score: number;
  matched_skills: string[];
  missing_skills: string[];
  uncertain_skills: string[];
  evidence: { skill: string; quote: string; source: string }[];
  explanation: string;
  status: ReviewStatus;
  questions: string[];
  signals: Record<string, number>;
}
export interface Pipeline {
  engine: string;
  label: string;
  ml1_available: boolean;
  ml2_available: boolean;
}
export interface Shortlist {
  vacancy: Vacancy;
  candidates: RankedCandidate[];
  pipeline: Pipeline;
  elapsed_ms: number;
}
export interface Match {
  vacancy: Vacancy;
  score: number;
  skill_score: number;
  matched_skills: string[];
  missing_skills: string[];
  explanation: string;
}
export interface Health {
  status: string;
  engine: string;
  ml1_available: boolean;
  ml2_available: boolean;
  vacancy_count: number;
  catalogue_note: string;
}
export interface Stats {
  candidate_count: number;
  shortlisted_count: number;
  review_count: number;
  feedback_count: number;
  custom_vacancy_count: number;
}
export interface VacancyPage {
  items: Vacancy[];
  total: number;
  page: number;
  limit: number;
  cities: string[];
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  const response = await fetch(`/api${path}`, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  if (!response.ok) {
    let detail = "Не удалось выполнить запрос. Попробуйте ещё раз.";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail))
        detail = data.detail
          .map((d: { msg?: string }) => d.msg || "Проверьте поля формы")
          .join("; ");
    } catch {
      /* server may be offline */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}
export const json = (value: unknown) => JSON.stringify(value);
export const scoreLabel = (score: number) =>
  (score >= 0 ? "+" : "") + score.toFixed(2);
export const statusLabel: Record<ReviewStatus, string> = {
  new: "На рассмотрении",
  shortlisted: "В шорт-листе",
  hold: "Вернуться позже",
};
export function safeSourceUrl(value: string) {
  try {
    const u = new URL(value);
    return ["https:", "http:"].includes(u.protocol) ? u.toString() : undefined;
  } catch {
    return undefined;
  }
}
