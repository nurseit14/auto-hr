import { scoreLabel } from "./api";
import type {
  ScoreBreakdownData,
  ScoreContribution,
  ScoreEvidence,
} from "./api";

const sourceLabels: Record<ScoreEvidence["source"], string> = {
  resume: "Резюме",
  clarification: "Ответ кандидата · самоотчёт",
  hr: "Подтверждение HR",
  vacancy: "Требования вакансии",
};
const mainFeatures = [
  "required_skill_coverage",
  "demonstrated_skill_coverage",
  "skill_coverage",
  "mandatory_skill_confirmation",
  "role_compatibility",
  "experience_compatibility",
  "relevant_experience_fulfillment",
  "seniority_compatibility",
  "task_coverage",
];
const number = (value: number, digits = 3) =>
  new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value);
const signed = (value: number, digits = 3) =>
  (value > 0 && Math.abs(value) >= 0.5 * 10 ** -digits ? "+" : "") +
  number(value, digits);

function years(value: number | null, required = false) {
  if (value === null) return "Не подтверждён";
  if (required && value === 0) return "Без минимума";
  const suffix = new Intl.PluralRules("ru-RU").select(value);
  const unit = required
    ? suffix === "one" || suffix === "other"
      ? "года"
      : "лет"
    : suffix === "one"
      ? "год"
      : suffix === "many"
        ? "лет"
        : "года";
  const count = new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: 2,
  }).format(value);
  return `${required ? "От " : ""}${count} ${unit}`;
}

function Evidence({ items }: { items: ScoreEvidence[] }) {
  if (!items.length) return null;
  return (
    <details className="score-evidence">
      <summary>Основания ({items.length})</summary>
      {items.map((item, index) => (
        <blockquote key={`${item.source}-${index}`}>
          <p>{item.quote}</p>
          <cite>{sourceLabels[item.source]}</cite>
        </blockquote>
      ))}
    </details>
  );
}

function Contribution({ item }: { item: ScoreContribution }) {
  const direction =
    item.contribution > 0
      ? "positive"
      : item.contribution < 0
        ? "negative"
        : "neutral";
  return (
    <li className="score-contribution">
      <div className="score-contribution-heading">
        <strong>{item.label}</strong>
        <span
          className={`score-contribution-value ${direction}`}
          title={`Вклад в балл: ${signed(item.contribution, 6)}`}
        >
          {signed(item.contribution)}
        </span>
      </div>
      <p>{item.reason}</p>
      {item.weight === 0 && (
        <p className="score-inactive">
          В линейной части модели у этого признака нулевой вес.
        </p>
      )}
      <Evidence items={item.evidence} />
    </li>
  );
}

export function ScoreBreakdown({
  data,
  compact = false,
}: {
  data: ScoreBreakdownData | null | undefined;
  compact?: boolean;
}) {
  if (!data) return null;
  const primary = [
    ...mainFeatures.flatMap((name) =>
      data.contributions.filter((item) => item.feature === name),
    ),
    ...data.contributions.filter(
      (item) => !mainFeatures.includes(item.feature) && item.contribution !== 0,
    ),
  ];
  const other = data.contributions.filter(
    (item) => !mainFeatures.includes(item.feature) && item.contribution === 0,
  );
  const adjustments = data.adjustments || [];
  const hasAdjustment = adjustments.some((item) => item.contribution !== 0);
  const adjustmentTotal = adjustments.reduce(
    (sum, item) => sum + item.contribution,
    0,
  );
  const version = data.model_version.match(/-v(\d+)$/)?.[1];
  return (
    <section
      className={`score-breakdown${compact ? " score-breakdown-compact" : ""}`}
      aria-label="Из чего сложился балл"
    >
      <div className="score-breakdown-heading">
        <h3>Из чего сложился балл</h3>
        <span className="score-version">
          {version ? `ML2 · v${version}` : "Подбор"}
        </span>
      </div>
      <p className="score-breakdown-intro">
        Каждая строка показывает свой вклад в итог. Плюс и минус помогают
        сравнивать результаты; это не решение о найме.
      </p>
      {data.warnings.length > 0 && (
        <ul className="score-warnings">
          {Array.from(new Set(data.warnings)).map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <dl className="score-experience">
        <div>
          <dt>Общий стаж</dt>
          <dd>{years(data.experience.general_years)}</dd>
        </div>
        <div>
          <dt>По этой роли</dt>
          <dd>{years(data.experience.relevant_years)}</dd>
        </div>
        <div>
          <dt>Требуется</dt>
          <dd>
            {data.experience.required_years === null
              ? "Не указан"
              : years(data.experience.required_years, true)}
          </dd>
        </div>
      </dl>
      <p className="score-context-reason">{data.experience.reason}</p>
      {data.role.reason && (
        <p className="score-context-reason">{data.role.reason}</p>
      )}
      <Evidence items={data.experience.evidence} />
      <ul className="score-contributions">
        {data.base_score !== 0 && (
          <li className="score-contribution">
            <div className="score-contribution-heading">
              <strong>Базовый уровень модели</strong>
              <span className="score-contribution-value">
                {signed(data.base_score)}
              </span>
            </div>
            <p>Постоянная часть расчёта, одинаковая для всех сравниваемых пар.</p>
          </li>
        )}
        {primary.map((item) => (
          <Contribution item={item} key={item.feature} />
        ))}
      </ul>
      {other.length > 0 && (
        <details className="score-other">
          <summary>Другие признаки без вклада ({other.length})</summary>
          <ul className="score-contributions">
            {other.map((item) => (
              <Contribution item={item} key={item.feature} />
            ))}
          </ul>
        </details>
      )}
      {adjustments.length > 0 && (
        <div className="score-rule-adjustments">
          {hasAdjustment && (
            <div className="score-model-subtotal">
              <span>Балл до правил вакансии</span>
              <strong>{scoreLabel(data.raw_score)}</strong>
            </div>
          )}
          <h4>Правила требований вакансии</h4>
          <p>
            Эти поправки применяются после суммы обученных признаков.
            Они показаны отдельно от коэффициентов модели.
          </p>
          <ul className="score-contributions">
            {adjustments.map((item) => (
              <li className="score-contribution" key={item.policy}>
                <div className="score-contribution-heading">
                  <strong>{item.label}</strong>
                  <span
                    className="score-contribution-value"
                    title={`Поправка к баллу: ${signed(item.contribution, 6)}`}
                  >
                    {signed(item.contribution)}
                  </span>
                </div>
                <p>{item.reason}</p>
                <Evidence items={item.evidence} />
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="score-breakdown-total">
        <strong>Итоговый балл</strong>
        <strong>{scoreLabel(data.final_score)}</strong>
      </div>
      <details className="score-calculation">
        <summary>Полный расчёт и коэффициенты</summary>
        <p>
          Вклады ниже округлены до шести знаков. Балл в списке округлён до двух.
        </p>
        <p>Вклад = (значение − среднее) ÷ масштаб × вес.</p>
        <div
          className="score-calculation-scroll"
          tabIndex={0}
          aria-label="Все признаки и их вклады, таблицу можно прокручивать"
        >
          <table>
            <caption>Все признаки модели</caption>
            <thead>
              <tr>
                <th scope="col">Признак</th>
                <th scope="col">Значение</th>
                <th scope="col">Среднее</th>
                <th scope="col">Масштаб</th>
                <th scope="col">Вес</th>
                <th scope="col">Вклад</th>
              </tr>
            </thead>
            <tbody>
              {data.contributions.map((item) => (
                <tr key={item.feature}>
                  <th scope="row">{item.label}</th>
                  <td>{number(item.value, 6)}</td>
                  <td>{number(item.mean, 6)}</td>
                  <td>{number(item.scale, 6)}</td>
                  <td>{number(item.weight, 6)}</td>
                  <td>{signed(item.contribution, 6)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row" colSpan={5}>Сумма модели с базовым уровнем</th>
                <td>{signed(data.raw_score, 6)}</td>
              </tr>
              {adjustments.map((item) => (
                <tr key={item.policy}>
                  <th scope="row" colSpan={5}>{item.label} · правило вакансии</th>
                  <td>{signed(item.contribution, 6)}</td>
                </tr>
              ))}
              <tr>
                <th scope="row" colSpan={5}>Итоговый балл</th>
                <td>{signed(data.final_score, 6)}</td>
              </tr>
            </tfoot>
          </table>
        </div>
        <p className="score-calculation-equation">
          Базовый уровень {signed(data.base_score, 6)} + вклады признаков ={" "}
          {signed(data.raw_score, 6)}. Поправки вакансии: {signed(adjustmentTotal, 6)}.
          {" "}Итог: {signed(data.raw_score, 6)} + ({signed(adjustmentTotal, 6)}) ={" "}
          {signed(data.final_score, 6)}
        </p>
        <p className="score-model-version">
          Модель: {data.model_version}. Признаки: {data.feature_version}.
        </p>
      </details>
    </section>
  );
}
