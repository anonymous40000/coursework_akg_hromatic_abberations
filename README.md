# Коррекция хроматических аберраций

Убирает поперечную (латеральную) хроматическую аберрацию — цветные ореолы по краям кадра. Работает в двух режимах: по калибровочному кадру точечной мишени и «слепой» — по самому снимку, без калибровки.

## Что нужно

Python 3.9+ и пакеты: numpy, opencv-python-headless, Pillow, pytest.

## Установка

```sh
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python-headless Pillow pytest
```

## Запуск

```sh
.venv/bin/python app.py
```

В окне: «Загрузить снимок» → выбрать файл → дождаться обработки. Результаты пишутся в `results/` рядом с исходником по имени файла: `<имя>_before.png`, `<имя>_after.png`, `<имя>_diff.png` (сравнение до/после). Журнал — `app.log`.

## Тесты

```sh
.venv/bin/python -m pytest tests/ -q
```

Ожидается `13 passed`.
