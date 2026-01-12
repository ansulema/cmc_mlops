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
- MLflow эксперименты: `mlruns/`

### DVC Remote
Удаленное хранилище (Google Drive, Яндекс Диск) не настроено из-за технических ограничений.
Данные нужно скачать скриптом `download_data.py`.

## Быстрый старт
```bash
git clone https://github.com/ansulema/cmc_mlops.git
cd cmc_mlops
git checkout hw2
pip install -r requirements.txt

# Скачать данные с HuggingFace
python download_data.py

# Запустить пайплайн
dvc repro
```

## Docker

### Сборка образа
```bash
# Убедитесь что модель обучена (models/spam_classifier/)
docker build -t ml-app:v1 .
```

### Запуск инференса
```bash
# Подготовить входной файл input.csv с колонкой 'text'
docker run --rm \
  -v $(pwd)/input.csv:/app/input.csv \
  -v $(pwd)/output:/app/output \
  ml-app:v1 \
  --input_path /app/input.csv \
  --output_path /app/output/preds.csv
```

### Формат данных

**Входной файл (input.csv):**
```csv
text
"Hello, how are you?"
"WIN FREE PRIZE NOW! Call 123456"
```

**Выходной файл (preds.csv):**
```csv
label,class_id,confidence,prob_ham,prob_spam
ham,0,0.95,0.95,0.05
spam,1,0.92,0.08,0.92
```

### Параметры скрипта
- `--input_path` — путь к входному CSV
- `--output_path` — путь к выходному CSV
- `--text_column` — имя колонки с текстом (по умолчанию: text)
- `--threshold` — порог классификации (по умолчанию: 0.5)

## TorchServe (онлайн-сервис)

### Сборка MAR-архива
```bash
cd torchserve
./build_mar.sh
```

### Сборка Docker образа
```bash
cd torchserve
docker build -t spam-serve:v1 .
```

### Запуск сервиса
```bash
docker run -d -p 8080:8080 -p 8081:8081 --name spam-serve spam-serve:v1
```

### REST API

**Prediction endpoint:**
```bash
curl -X POST http://localhost:8080/predictions/spam_classifier \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won $1000000!"}'
```

**Ответ:**
```json
{
  "label": "spam",
  "class_id": 1,
  "confidence": 0.92,
  "prob_ham": 0.08,
  "prob_spam": 0.92
}
```

**Health check:**
```bash
curl http://localhost:8080/ping
```

**Model info:**
```bash
curl http://localhost:8081/models/spam_classifier
```

### Конфигурация (torchserve/config.properties)
- `inference_address` — адрес для предсказаний (порт 8080)
- `management_address` — адрес управления (порт 8081)
- `batchSize` — размер батча (8)
- `maxBatchDelay` — макс. задержка батча в мс (100)
- `responseTimeout` — таймаут ответа в сек (120)

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
2. **train** — fine-tuning DistilBERT + MLflow tracking
3. **evaluate** — метрики на тестовой выборке -> `metrics.json`

## MLflow

### Просмотр результатов
```bash
mlflow ui
# Открыть http://localhost:5000
```

### Что логируется
- **Параметры:** model_name, learning_rate, batch_size, num_epochs, и др.
- **Метрики:** accuracy, f1_spam, recall_spam, fpr
- **Артефакты:** config.yaml, dvc.lock
- **Теги:** dvc_data_hash (хеш датасета)

## Структура проекта
```
Dockerfile           # Docker образ для batch инференса
config.yaml          # параметры обучения
dvc.yaml             # DVC pipeline
dvc.lock             # зафиксированные версии
mlruns/              # MLflow эксперименты

prepare.py           # подготовка данных
train.py             # обучение модели + MLflow
evaluate.py          # оценка модели + MLflow
inference.py         # инференс + интерактивный режим
download_data.py     # скачивание датасета с HF

src/
  predict.py         # batch prediction для Docker

torchserve/          # TorchServe online service
  handler.py         # кастомный обработчик
  Dockerfile         # образ на базе pytorch/torchserve
  config.properties  # конфигурация сервиса
  build_mar.sh       # скрипт сборки .mar архива
  model-store/       # MAR архивы (генерируется)

data/
  russian_spam.csv   # сырые данные (DVC)
  train.csv          # train split
  val.csv            # validation split
  test.csv           # test split

models/
  spam_classifier/   # обученная модель

tests/               # unit тесты
.github/workflows/   # CI: автозапуск тестов
```

## Тестирование
```bash
cd tests && pytest -v
```
