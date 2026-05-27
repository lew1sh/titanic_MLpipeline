# Titanic ML Pipeline

Проект решает задачу бинарной классификации из соревнования Titanic: по данным о пассажире нужно предсказать, выжил он или нет.

Основная цель репозитория — сделать воспроизводимый ML-пайплайн: установить зависимости, запустить один скрипт и получить таблицу результатов моделей и submission-файл лучшей модели.

## Что реализовано

- EDA вынесен в отдельный ноутбук: `notebooks/eda.ipynb`.
- Основной пайплайн запускается через `main.py`.
- Настройки путей, кросс-валидации, групп моделей и гиперпараметров лежат в `config.yaml`.
- Повторно используемый код разнесен по модулям в `src/`.
- Есть baseline-модели: логистическая регрессия, KNN, дерево решений, Random Forest.
- Есть несколько boosting-моделей: CatBoost, LightGBM, XGBoost.
- Есть ансамбли: Voting и Stacking.
- Есть простая DL-модель для табличных данных: `MLPClassifier`.
- Результаты сохраняются в `outputs/`.

## Структура проекта

```text
titanic_MLpipeline/
  data/
    train.csv
    test.csv
    gender_submission.csv

  notebooks/
    eda.ipynb

  src/
    config.py        # загрузка config.yaml
    data.py          # чтение train/test
    features.py      # feature engineering
    preprocessing.py # кодирование категориальных признаков
    evaluation.py    # сбор таблицы метрик
    models.py        # обучение и CV разных групп моделей
    submission.py    # сохранение submission-файлов

  config.yaml
  main.py
  requirements.txt
  README.md
```

## Данные

Используются стандартные файлы Titanic:

- `data/train.csv` — обучающая выборка с таргетом `Survived`;
- `data/test.csv` — тестовая выборка без таргета;
- `data/gender_submission.csv` — пример формата сабмита.

Идентификатор `PassengerId` используется как индекс.

## EDA

Исследовательский анализ находится в `notebooks/eda.ipynb`.

В ноутбуке проверяются:

- общая структура train/test;
- пропуски в `Age`, `Cabin`, `Embarked`;
- распределения числовых признаков;
- выбросы в `Fare`, `Parch`, `SibSp`;
- связь некоторых признаков с `Survived`;
- корреляции между числовыми признаками.

Ключевые выводы из EDA используются в feature engineering:

- из `Cabin` создается бинарный признак `HasCabin`;
- из `Name` извлекается титул пассажира `Title`;
- пропуски в `Age` заполняются средним значением;
- пропуски в `Embarked` заполняются модой;
- `Fare` логарифмируется через `log1p`.

## Запуск

Создать и активировать окружение можно любым удобным способом. Например:

```bash
python -m venv .venv
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить основной пайплайн:

```bash
python main.py
```

После запуска появятся новые файлы:

```text
outputs/results.csv
outputs/submission_<best_model_name>.csv
```

`results.csv` содержит среднее качество и стандартное отклонение на Stratified K-Fold CV. Submission-файл создается только для лучшей модели из этой таблицы.

## Конфигурация

Основные настройки лежат в `config.yaml`.

Пример:

```yaml
validation:
  n_splits: 5
  shuffle: true
  scoring: accuracy

models:
  baselines:
    enabled: true
  boosting:
    enabled: true
  ensembles:
    enabled: true
  mlp:
    enabled: true
```

Чтобы отключить группу моделей, достаточно поменять `enabled`:

```yaml
models:
  ensembles:
    enabled: false
```

Гиперпараметры моделей меняются в блоке `model_params`:

```yaml
model_params:
  catboost:
    iterations: 300
    depth: 4
    learning_rate: 0.05

  xgboost:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 4

  mlp:
    hidden_layer_sizes:
      - 64
      - 32
    activation: relu
    alpha: 0.0001
```

## Модели

В проекте сравниваются несколько групп моделей.

Baseline:

- Logistic Regression без регуляризации;
- Logistic Regression с L1/L2/ElasticNet;
- KNN;
- Decision Tree;
- Random Forest.

Boosting:

- CatBoost;
- LightGBM;
- XGBoost.

Ensembles:

- hard voting;
- soft voting;
- stacking с Logistic Regression;
- stacking с RidgeClassifier.

DL baseline:

- `MLPClassifier` с двумя скрытыми слоями.

## Метрика

Сейчас используется `accuracy`, так как это базовая метрика для Titanic и она легко интерпретируется.

Кросс-валидация:

- `StratifiedKFold`;
- 5 фолдов;
- `random_state: 42`.

## Результаты

Итоговая таблица сохраняется в:

```text
outputs/results.csv
```

В ней будут колонки:

- `model`;
- `mean_accuracy`;
- `std_accuracy`.

Текущий лучший результат после запуска пайплайна:

| Model | Mean accuracy | Std accuracy |
|---|---:|---:|
| CatBoost | 0.8406 | 0.0216 |

Финальный submission-файл сохраняется в:

```text
outputs/submission_<best_model_name>.csv
```

Например, если лучшей моделью окажется CatBoost, файл будет называться `outputs/submission_catboost.csv`.


