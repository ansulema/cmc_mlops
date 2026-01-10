# Spam Classifier

## Бизнес-цель
Автоматическая фильтрация спам-сообщений для защиты пользователей от мошенничества и нежелательного контента. 

**Почему ML:** Правила и ключевые слова легко обходятся спамерами. Нейросеть обучается на паттернах и обобщает, что позволяет ловить новые виды спама.

## Целевые метрики

### Бизнес-метрики
- **Recall (spam)** >= 0.85 — не пропускать спам пользователям
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

## Установка
```bash
pip install -r requirements.txt
```

## Использование
```bash
# Скачать данные
python download_data.py

# Обучение
python train.py --config config.yaml

# Инференс
python inference.py --model models/spam_classifier
```

## Структура проекта
```
config.yaml          # параметры обучения (lr, batch_size, epochs, seed)
train.py             # скрипт обучения с логированием
inference.py         # инференс + интерактивный режим
download_data.py     # скачивание датасета

src/
  data.py            # валидация и предобработка данных
  predict.py         # логика предсказаний (logits -> probs -> class)

tests/
  test_data.py       # тесты валидации и предобработки
  test_predict.py    # тесты перевода предсказаний

.github/workflows/
  tests.yml          # CI: автозапуск тестов при коммите
```

## Тестирование
```bash
cd tests && pytest -v
```

Тесты проверяют:
- Валидацию структуры данных (колонки, типы, null)
- Корректность меток (0/1)
- Предобработку (маппинг, дубликаты, сплит)
- Перевод logits -> probs -> class -> label
- Граничные случаи (порог 0.5, невалидные классы)

## CI/CD
При каждом push/PR в ветки `main`, `master`, `hw1` автоматически запускаются тесты через GitHub Actions.
