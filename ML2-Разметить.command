#!/bin/zsh
cd "${0:A:h}" || exit 1
./.venv-label-studio/bin/python -m ML2.run --publish
result=$?
echo ""
if [ "$result" -eq 0 ]; then
  echo "Готово. Ссылка на проект Label Studio указана выше."
else
  echo "Обработка завершилась с ошибкой. Сообщение указано выше; готовые ответы сохранены."
fi
read -r "?Нажми Enter, чтобы закрыть окно..."
exit "$result"
