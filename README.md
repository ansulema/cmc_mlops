# Spam Classifier

## Бизнес-цель
Автоматическая фильтрация спам-сообщений для защиты пользователей от мошенничества и нежелательного контента.

**Почему ML:** Правила и ключевые слова легко обходятся спамерами. Нейросеть обучается на паттернах и обобщает, что позволяет ловить новые виды спама.

## Целевые метрики

### Бизнес-метрики
- **Recall (spam)** >= 0.85 — не пропускать спам
- **FPR (ham->spam)** <= 1% — не блокировать нормальные сообщения

### ML-метрики (на test)
- **F1-score (spam)** >= 0.9
- **Accuracy** >= 0.95

### Технические метрики (инференс)
- p95 latency <= 200 мс (CPU, batch=1)
- Доля 5xx/timeouts <= 1%

## Данные
**Russian Spam Detection Dataset**: https://huggingface.co/datasets/darkQibit/russian-spam-detection  
- ~4.5M примеров (92% ham, 8% spam)
- Колонки: `message`, `label` (0/1)

### Где лежат данные/модели
- Сырые данные: `data/russian_spam.csv` (под DVC)
- Подготовленные: `data/train.csv`, `data/val.csv`, `data/test.csv`
- Модель: `models/spam_classifier/` (генерируется пайплайном)
- DVC remote: локальное хранилище (можно заменить на GDrive/S3)

## Быстрый старт
```bash
git clone https://github.com/ansulema/cmc_mlops.git
cd cmc_mlops
pip install -r requirements.txt
dvc pull
dvc repro
```

## DVC Pipeline
```bash
# Запуск всего пайплайна
dvc repro

# Отдельные стейджи
dvc repro prepare   # предобработка данных
dvc repro train     # обучение модели
dvc repro evaluate  # оценка на тесте
```

### Стейджи пайплайна
1. **prepare** — загрузка, предобработка, split на train/val/test
2. **train** — fine-tuning DistilBERT
3. **evaluate** — метрики на тестовой выборке -> `metrics.json`

## Структура проекта
```
config.yaml          # параметры обучения
dvc.yaml             # DVC pipeline
dvc.lock             # зафиксированные версии

prepare.py           # подготовка данных
train.py             # обучение модели  
evaluate.py          # оценка модели
inference.py         # инференс + интерактивный режим
download_data.py     # скачивание датасета с HF

data/
  russian_spam.csv   # сырые данные (DVC)
  train.csv          # train split
  val.csv            # validation split
  test.csv           # test split

models/
  spam_classifier/   # обученная модель

src/                 # модули для тестов
tests/               # unit тесты

.github/workflows/   # CI: автозапуск тестов
```

## Версионирование
```bash
# Сохранить текущую версию
dvc push

# Восстановить версию
git checkout <commit>
dvc pull

# Переключение между версиями данных/моделей
dvc checkout
```

## Тестирование
```bash
cd tests && pytest -v
```
