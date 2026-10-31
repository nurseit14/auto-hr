#!/usr/bin/env python3
"""Prepare the local MVP and replace this process with the application server."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def fresh(marker: Path, value: str) -> bool:
    return marker.exists() and marker.read_text(encoding="utf-8").strip() == value


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный запуск Auto-HR")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUTO_HR_PORT", "8000")))
    parser.add_argument("--skip-install", action="store_true", help="Не устанавливать зависимости повторно")
    parser.add_argument("--no-build", action="store_true", help="Использовать уже собранный frontend/dist")
    parser.add_argument("--prepare-only", action="store_true", help="Установить зависимости и собрать, не запускать сервер")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("Нужен Python 3.12 или новее.")
    if not 1 <= args.port <= 65535:
        raise RuntimeError("Порт должен быть от 1 до 65535.")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    node = shutil.which("node")
    if not npm or not node:
        raise RuntimeError("Нужны Node.js 22 LTS или новее и npm. Установите их и повторите запуск.")
    node_version = tuple(int(part) for part in subprocess.check_output([node, "--version"], text=True).strip().lstrip("v").split(".")[:2])
    if node_version < (22, 12):
        raise RuntimeError("Нужен Node.js 22.12 или новее.")
    if not PYTHON.exists():
        print("Создаю изолированное окружение Python…", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    if not args.skip_install:
        requirement = ROOT / "backend/requirements.lock"
        dependency_hash = fingerprint([requirement])
        backend_marker = VENV / ".auto_hr_requirements.sha256"
        if not fresh(backend_marker, dependency_hash):
            print("Устанавливаю зависимости бэкенда…", flush=True)
            # Environments created by tools such as uv may intentionally omit pip.
            pip_probe = subprocess.run(
                [str(PYTHON), "-m", "pip", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if pip_probe.returncode:
                run([str(PYTHON), "-m", "ensurepip", "--upgrade"])
            run([str(PYTHON), "-m", "pip", "install", "-r", str(requirement)])
            backend_marker.write_text(dependency_hash + "\n", encoding="utf-8")
        package_inputs = [FRONTEND / "package.json", FRONTEND / "package-lock.json"]
        frontend_marker = FRONTEND / "node_modules/.auto_hr_package.sha256"
        package_hash = fingerprint(package_inputs)
        if not fresh(frontend_marker, package_hash):
            print("Устанавливаю зависимости фронтенда…", flush=True)
            install = "ci" if (FRONTEND / "package-lock.json").exists() else "install"
            run([npm, install, "--no-audit", "--no-fund"], cwd=FRONTEND)
            frontend_marker.write_text(fingerprint(package_inputs) + "\n", encoding="utf-8")
    if not args.no_build:
        print("Собираю интерфейс…", flush=True)
        run([npm, "run", "build"], cwd=FRONTEND)
    if not (FRONTEND / "dist/index.html").exists():
        raise RuntimeError("Нет frontend/dist/index.html. Запустите без флага --no-build.")
    if args.prepare_only:
        print("Готово. Для запуска: ./start.command --skip-install --no-build", flush=True)
        return
    os.chdir(ROOT)
    os.environ.setdefault("AUTO_HR_DB", str(ROOT / "backend/runtime/auto_hr.sqlite3"))
    print(f"\nAuto-HR: http://127.0.0.1:{args.port}\nОстановить сервер: Ctrl+C\n", flush=True)
    os.execv(str(PYTHON), [str(PYTHON), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(args.port)])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗапуск остановлен.", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"\nНе удалось запустить Auto-HR: {error}", file=sys.stderr)
        print("Проверьте доступ к интернету при первой установке и версии Python/Node.js.", file=sys.stderr)
        sys.exit(1)
