import { useEffect, useRef, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { advanceAuthGeneration, api } from "./api";
import type { AuthUser } from "./api";
import HRWorkspace from "./App";
import { CandidateWorkspace } from "./CandidateWorkspace";
import { LoginPage } from "./LoginPage";

export default function AuthApp() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [loggingOut, setLoggingOut] = useState(false);
  const sessionRequest = useRef<AbortController | null>(null);
  const sessionRevision = useRef(0);
  const currentAccount = useRef("");
  function invalidateSessionRequests() {
    sessionRevision.current += 1;
    sessionRequest.current?.abort();
    sessionRequest.current = null;
  }
  function acceptUser(value: AuthUser, broadcast = true) {
    invalidateSessionRequests();
    const account = `${value.username}:${value.role}`;
    if (currentAccount.current !== account) advanceAuthGeneration();
    currentAccount.current = account;
    if (value.role === "hr") {
      const legacy = localStorage.getItem("autohr_vacancy");
      if (legacy && !localStorage.getItem("autohr_vacancy_hr")) localStorage.setItem("autohr_vacancy_hr", legacy);
      localStorage.removeItem("autohr_vacancy");
    }
    setError("");
    setLoading(false);
    setUser(value);
    if (broadcast) localStorage.setItem("autohr_auth_change", String(Date.now()));
  }
  useEffect(() => {
    const controller = new AbortController();
    invalidateSessionRequests();
    sessionRequest.current = controller;
    const revision = sessionRevision.current;
    setLoading(true);
    setError("");
    api<{ user: AuthUser | null }>("/auth/me", { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted && revision === sessionRevision.current) { if (result.user) acceptUser(result.user, false); else { currentAccount.current = ""; setUser(null); setLoading(false); } } })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Не удалось подключиться к Auto-HR."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [retry]);
  useEffect(() => {
    const expired = () => { invalidateSessionRequests(); advanceAuthGeneration(); currentAccount.current = ""; setUser(null); setLoading(false); setError(""); };
    let active = true;
    const revalidate = (blocking = false) => {
      invalidateSessionRequests();
      const controller = new AbortController();
      sessionRequest.current = controller;
      const revision = sessionRevision.current;
      if (blocking) { advanceAuthGeneration(); currentAccount.current = ""; setUser(null); setLoading(true); }
      api<{ user: AuthUser | null }>("/auth/me", { signal: controller.signal }).then(result => {
        if (!active || controller.signal.aborted || revision !== sessionRevision.current) return;
        if (result.user) acceptUser(result.user, false); else { advanceAuthGeneration(); currentAccount.current = ""; setUser(null); setLoading(false); }
      }).catch(reason => {
        if (!active || controller.signal.aborted || revision !== sessionRevision.current) return;
        setLoading(false);
        if (blocking || !currentAccount.current) setError(reason instanceof Error ? reason.message : "Не удалось подключиться к Auto-HR.");
      });
    };
    const focus = () => revalidate();
    const visible = () => { if (document.visibilityState === "visible") revalidate(); };
    const storage = (event: StorageEvent) => { if (event.key === "autohr_auth_change") revalidate(true); };
    window.addEventListener("autohr:unauthorized", expired);
    window.addEventListener("focus", focus);
    window.addEventListener("storage", storage);
    document.addEventListener("visibilitychange", visible);
    return () => { active = false; invalidateSessionRequests(); window.removeEventListener("autohr:unauthorized", expired); window.removeEventListener("focus", focus); window.removeEventListener("storage", storage); document.removeEventListener("visibilitychange", visible); };
  }, []);
  async function logout() {
    if (loggingOut) return;
    invalidateSessionRequests();
    advanceAuthGeneration();
    setLoggingOut(true);
    setError("");
    try {
      await api("/auth/logout", { method: "POST" });
      if (user) localStorage.removeItem(`autohr_vacancy_${user.role}`);
      invalidateSessionRequests();
      currentAccount.current = "";
      setUser(null);
      localStorage.setItem("autohr_auth_change", String(Date.now()));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось выйти. Попробуйте ещё раз.");
    } finally { setLoggingOut(false); }
  }
  if (loading) return <div className="auth-loading" role="status"><LoaderCircle size={28} className="spin" /><p>Открываем Auto-HR…</p></div>;
  if (!user && error) return <div className="auth-loading"><h1>Не удалось подключиться</h1><p role="alert">{error}</p><button className="button primary" onClick={() => setRetry(value => value + 1)}>Повторить</button></div>;
  if (!user) return <LoginPage onLogin={acceptUser} />;
  return <>{error && <div className="auth-global-error" role="alert">{error}<button className="text-button" onClick={() => setError("")}>Закрыть</button></div>}{user.role === "hr" ? <HRWorkspace key={`${user.username}:hr`} user={user} onLogout={logout} loggingOut={loggingOut} /> : <CandidateWorkspace key={`${user.username}:candidate`} user={user} onLogout={logout} loggingOut={loggingOut} />}</>;
}
