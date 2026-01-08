# Spam Classifier

## Цель проекта
Классификация сообщений на spam/ham (fine-tuning DistilBERT) для автоматической фильтрации спама.

## Целевые метрики
### Качество модели (на test)
- **F1-score (spam)** >= 0.9
- **Recall (spam)** >= 0.85
- **FPR (ham->spam)** <= 1%

### Сервис (инференс)
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
python inference.py --model ./spam_classifier
```

## Структура
- `config.yaml` — параметры обучения
- `train.py` — скрипт обучения
- `inference.py` — примеры + интерактивный режим
- `download_data.py` — скачивание датасета
