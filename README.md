# SMS Spam Classifier

## Цель проекта
Построить модель классификации SMS на `spam/ham` по тексту (fine-tuning HF Transformers), чтобы автоматически фильтровать спам-сообщения и минимизировать ложные блокировки “хороших” сообщений.

## Целевые метрики для продакшена
### Сервис (инференс)
- **p95 время отклика** ≤ **200 мс** (CPU, batch=1)
- **Доля неуспешных запросов** (5xx/timeouts) ≤ **1%**
- **Ресурсы**: RAM ≤ **1 ГБ**, CPU ≤ **1 vCPU** (или в рамках выделенного SLA окружения)

### Качество модели (offline, на test)
- **Accuracy** ≥ **0.95**
- **F1-score (spam)** ≥ **0.93**
- **Recall (spam)** ≥ **0.95** (не пропускать спам)
- **False Positive Rate (ham→spam)** ≤ **1%** (не блокировать нормальные SMS)

## Набор данных
**SMS Spam Collection Dataset (Kaggle)**  
Источник: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
Поля:
- `v1` — метка (`ham`/`spam`)
- `v2` — текст сообщения  
Подготовка:
- переименовать в `label`, `text`
- удалить пустые/дубликаты (если есть)
- split: train/val/test = 80/10/10 со стратификацией по `label`

## План экспериментов
1) **Baseline**: TF-IDF + Logistic Regression.  
   Метрики: Accuracy, F1(spam), Recall(spam), FPR.

2) **HF fine-tuning**: модель на основе bert архитектуры.  
   Подбор: learning rate, batch size, epochs.

3) **Подбор порога** вероятности под ограничения на FPR/Recall.

4) **Финальная оценка** на test + сохранение модели в HF-формате (model + tokenizer).
