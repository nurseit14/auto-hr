import { useState } from "react";
import type { FormEvent } from "react";
import { ArrowRight, Eye, EyeOff, FileText, LoaderCircle, MessageSquare, Search } from "lucide-react";
import { api, json } from "./api";
import type { AuthUser } from "./api";

export function BrandMark() {
  return <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>;
}

export function LoginPage({ onLogin }: { onLogin: (user: AuthUser) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [visible, setVisible] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api<{ user: AuthUser }>("/auth/login", {
        method: "POST", body: json({ username: username.trim(), password }),
      });
      onLogin(result.user);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось войти. Попробуйте ещё раз.");
    } finally { setBusy(false); }
  }
  return <main className="login-page">
    <section className="login-intro" aria-label="Auto-HR">
      <div className="login-brand"><BrandMark /><span>Auto-HR</span></div>
      <div className="login-intro-copy">
        <h2>Вакансии, резюме и понятные основания подбора.</h2>
        <p>Рабочее пространство для IT-кандидатов и HR в Казахстане.</p>
        <ul className="login-purpose-list">
          <li><Search size={21} /><span>Подбор вакансий по вашему опыту</span></li>
          <li><FileText size={21} /><span>Навыки и подтверждения из резюме</span></li>
          <li><MessageSquare size={21} /><span>Обратная связь и объяснение балла</span></li>
        </ul>
      </div>
      <p className="login-intro-foot">Финальное решение о найме принимает человек.</p>
    </section>
    <section className="login-form-side">
      <div className="login-form-wrap">
        <h1>Вход в Auto-HR</h1>
        <p className="muted login-description">Введите логин и пароль. Откроется ваш кабинет.</p>
        <form className="stack login-form" onSubmit={submit} aria-busy={busy}>
          <label className="field">Логин
            <input required autoComplete="username" name="username" maxLength={120} autoFocus value={username} onChange={event => setUsername(event.target.value)} disabled={busy} placeholder="Введите логин" />
          </label>
          <label className="field">Пароль
            <span className="password-field">
              <input required autoComplete="current-password" name="password" type={visible ? "text" : "password"} maxLength={200} value={password} onChange={event => setPassword(event.target.value)} disabled={busy} placeholder="Введите пароль" />
              <button type="button" className="icon-button" aria-label={visible ? "Скрыть пароль" : "Показать пароль"} aria-pressed={visible} onClick={() => setVisible(value => !value)} disabled={busy}>{visible ? <EyeOff size={18} /> : <Eye size={18} />}</button>
            </span>
          </label>
          {error && <p className="form-error" role="alert">{error}</p>}
          <button className="button primary login-submit" disabled={busy}>{busy ? <LoaderCircle size={18} className="spin" /> : <ArrowRight size={18} />}{busy ? "Входим…" : "Войти"}</button>
        </form>
        <div className="login-access">
          <h2>Доступ для демонстрации</h2>
          <dl><div><dt>Кабинет HR</dt><dd>admin / admin</dd></div><div><dt>Кабинет кандидата</dt><dd>user / user</dd></div></dl>
          <p>Тип кабинета определяется по логину.</p>
        </div>
      </div>
    </section>
  </main>;
}
