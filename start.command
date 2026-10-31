#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен Python 3.12 или новее. Установите Python и повторите запуск."
  exit 1
fi
exec python3 scripts/start.py "$@"
