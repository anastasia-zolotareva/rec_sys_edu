# Гибридная рекомендательная система Dueling DQN и DeepFM+SVD++

## Описание проекта

Прототип представляет собой гибридную рекомендательную систему для формирования индивидуальных образовательных траекторий, объединяющую методы глубокого обучения, матричной факторизации и обучения с подкреплением.

### Решаемые задачи

1. **Многокритериальность и динамичность предпочтений** - одновременный учет образовательной ценности, сложности материала, личных интересов и меняющихся целей студентов
2. **Оптимизация функции кумулятивного вознаграждения** - формирование образовательных траекторий
3. **Проблема разреженности данных и коротких коммуникаций** - работа с неполными данными о студентах и их предпочтениях
4. **Адаптация к контексту** - учет специализации, семестра и внешних условий, влияющих на выбор учебного материала

### Компоненты системы

- **DeepFM+SVD++** - гибридная модель для предсказания многокритериальных рейтингов
- **Dueling DQN** - агент обучения с подкреплением для долгосрочного планирования
- **EducationalEnvironment** - параметры среды с учетом контекста

## Структура проекта

```
rec_sys_edu/
├── configs/                 # YAML-конфигурации ITM-Rec и OULAD
│   ├── itmrec.yaml          # Конфиг базового сценария (ITM-Rec)
│   └── oulad.yaml           # Конфиг расширенного сценария (OULAD)
├── src/                     # Исходный код
│   ├── api.py               # Унифицированный Python API (build_config, train_*, evaluate_system)
│   ├── cli.py               # Единая CLI (`python -m src.cli ...`)
│   ├── data/                # Модуль данных
│   │   ├── loaders.py       # Загрузка и диспетчер `load_dataset(type)`
│   │   ├── preprocess_itmrec.py  # ITM-Rec → DatasetBundle
│   │   ├── preprocess_oulad.py   # OULAD → DatasetBundle (+ mixed-step catalog)
│   │   ├── oulad_reports.py      # Таблицы и графики EDA для OULAD
│   │   ├── schemas.py            # DatasetBundle
│   │   └── dataset.py            # PyTorch Dataset (ITMDataset)
│   ├── models/
│   │   ├── deepfm_svdpp.py       # DeepFM+SVD++ с save/load_checkpoint
│   │   └── dueling_dqn.py        # Dueling DQN с action-mask
│   ├── environment/
│   │   ├── educational_env.py    # Среда ITM-Rec (reward, novelty, termination)
│   │   ├── oulad_env.py          # Среда OULAD (proxy-reward, terminal bonus)
│   │   ├── oulad_state.py        # Кодирование 96-мерного state OULAD
│   │   ├── action_mask.py        # Action mask OULAD (week, kind, completed)
│   │   ├── reward.py             # Централизованные функции наград
│   │   └── state_encoder.py      # ITM-Rec state encoder
│   ├── training/
│   │   ├── trainer.py            # DQNTrainer (mask-aware TD-target)
│   │   ├── train_static.py       # Пайплайн DeepFM+SVD++
│   │   ├── train_dqn.py          # Пайплайн DQN (ITM-Rec и OULAD)
│   │   ├── replay_buffer.py      # PrioritizedReplayBuffer
│   │   └── config.py             # ITMREC_DEFAULTS / OULAD_DEFAULTS
│   ├── evaluation/
│   │   ├── metrics.py                # Короткие и долгосрочные метрики
│   │   ├── adaptability.py           # AdaptabilityScore, Stability (H1)
│   │   ├── comparative_tester.py     # H2.1 vs baselines
│   │   ├── long_term_evaluator.py    # CDR, Retention, LearningSlope (H2)
│   │   ├── novelty_ablation.py       # Ablation no_context/no_demo/no_novelty (H3)
│   │   ├── statistics.py             # Welch, Wilcoxon, Cohen's d, bootstrap CI
│   │   ├── trajectory_visualizer.py  # Визуализация reward/coverage/novelty
│   │   └── system_evaluator.py       # Оркестратор H1/H2/H3
│   └── utils/
│       ├── helpers.py                # prepare_run, save_metrics, set_seed, device
│       └── visualization.py          # Вспомогательные графики
├── notebooks/               # Подробнее: notebooks/README.md
│   ├── 00_quickstart.ipynb           # Быстрый старт: полный цикл (данные → DeepFM → DQN → H1–H3 → визуализация)
│   ├── 01_data_analysis.ipynb        # EDA ITM-Rec
│   ├── 02_model_development.ipynb    # Разработка моделей (устаревший, см. 00 или src.api)
│   ├── 03_testing.ipynb              # Тестирование (устаревший, см. 06/07)
│   ├── 04_oulad_data.ipynb           # EDA и DatasetBundle для OULAD
│   ├── 05_oulad_model.ipynb          # Обучение DeepFM+DQN для OULAD
│   ├── 06_hypotheses.ipynb           # H1/H2/H3 через api.evaluate_system
│   └── 07_trajectories.ipynb         # Визуализация траекторий
├── scripts/                 # Тонкие обертки над CLI (для обратной совместимости)
│   ├── download_data.py
│   ├── prepare_oulad.py
│   ├── analyze_oulad.py
│   ├── train_deepfm.py
│   ├── train_dqn.py
│   └── evaluate_system.py
├── tests/                   # pytest-инварианты (state_dim, mask, reward, checkpoint)
│   ├── test_invariants.py
│   └── test_dqn_mask.py
├── schemes/                 # Схемы архитектуры (draw.io XML)
├── data/
│   ├── raw/                 # Исходные данные (data/raw/oulad для OULAD)
│   ├── processed/           # Обработанные данные + EDA OULAD
│   └── models/              # Чекпоинты DeepFM / DQN
├── results/                 # Результаты экспериментов (run-директории)
├── requirements.txt
└── pyproject.toml
```

## Установка и настройка

### Предварительные требования

- Python 3.9+
- CUDA (опционально, для GPU ускорения)

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка проекта в режиме разработки (для корректной работы импортов)
pip install -e .
```

### Настройка Kaggle API

Для загрузки данных с Kaggle необходимо настроить API:

1. Создать аккаунт на [Kaggle](https://www.kaggle.com/)
2. Перейти в Settings → API → Create New Token
3. Сохранить `kaggle.json` в:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Установить права доступа (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

## Быстрый старт

Прототип управляется единым CLI (`python -m src.cli ...`) и симметричным
Python API (`src.api`). Все ключевые сценарии - подготовка данных, EDA,
обучение, оценка гипотез - поддерживают оба датасета: **ITM-Rec** (базовый)
и **OULAD** (расширенный).

### 1. Подготовка данных

```bash
# ITM-Rec (Kaggle)
python -m src.cli data download --dataset itmrec
python -m src.cli data prepare  --dataset itmrec --config configs/itmrec.yaml

# OULAD (Open University Learning Analytics)
python -m src.cli data download --dataset oulad
python -m src.cli data prepare  --dataset oulad --config configs/oulad.yaml

# Разведочный анализ OULAD: CSV-таблицы + PNG-графики
python -m src.cli data analyze  --dataset oulad
```

### 2. Обучение

```bash
# DeepFM+SVD++ (multi-head: Rating/App/Data/Ease для ITM-Rec, proxy-критерии для OULAD)
python -m src.cli train static --dataset itmrec --config configs/itmrec.yaml
python -m src.cli train static --dataset oulad  --config configs/oulad.yaml

# Dueling DQN поверх DeepFM (action-mask, prioritized replay)
python -m src.cli train dqn --dataset itmrec \
    --deepfm-checkpoint data/models/deepfm_itmrec_best.pth
python -m src.cli train dqn --dataset oulad \
    --deepfm-checkpoint data/models/deepfm_oulad_best.pth
```

### 3. Оценка гипотез H1/H2/H3

```bash
python -m src.cli evaluate --dataset itmrec --hypothesis all \
    --deepfm-checkpoint data/models/deepfm_itmrec_best.pth \
    --dqn-checkpoint    data/models/dqn_itmrec_best.pth

python -m src.cli evaluate --dataset oulad --hypothesis H2 \
    --deepfm-checkpoint data/models/deepfm_oulad_best.pth \
    --dqn-checkpoint    data/models/dqn_oulad_best.pth
```

### 4. Использование ноутбуков

Рекомендуемая точка входа — `notebooks/00_quickstart.ipynb` (полный цикл за 10–15 минут; ITM-Rec или OULAD по конфигу). Детальные сценарии и описание API в каталоге см. [notebooks/README.md](notebooks/README.md).

```bash
jupyter notebook notebooks/00_quickstart.ipynb   # Быстрый старт (рекомендуется)
jupyter notebook notebooks/04_oulad_data.ipynb   # EDA + DatasetBundle OULAD
jupyter notebook notebooks/05_oulad_model.ipynb  # Обучение DeepFM+DQN для OULAD
jupyter notebook notebooks/06_hypotheses.ipynb   # Гипотезы H1/H2/H3 (оба датасета)
jupyter notebook notebooks/07_trajectories.ipynb # Визуализация траекторий
```

### 5. Python API (эквивалент CLI)

```python
from src import api

config = api.build_config("oulad", yaml_path="configs/oulad.yaml")
run_dir = api.prepare_run(config, run_name="oulad_experiment")

static = api.train_static("oulad", config=config, run_dir=run_dir)
dqn = api.train_dqn(
    "oulad",
    config=config,
    run_dir=run_dir,
    deepfm_checkpoint=static["history"]["best_checkpoint"],
)
results = api.evaluate_system(
    "oulad",
    hypothesis="all",
    config=config,
    run_dir=run_dir,
    deepfm_checkpoint=static["history"]["best_checkpoint"],
    dqn_checkpoint=dqn["checkpoint"],
)
```

### 6. Вспомогательные скрипты (обертки над CLI)

Старые вызовы `python scripts/*.py` продолжают работать:

```bash
python scripts/download_data.py --dataset oulad
python scripts/prepare_oulad.py --config configs/oulad.yaml
python scripts/analyze_oulad.py
python scripts/train_deepfm.py --dataset oulad
python scripts/train_dqn.py    --dataset oulad --episodes 200
python scripts/evaluate_system.py --dataset oulad --hypothesis all \
    --deepfm-checkpoint ... --dqn-checkpoint ...
```

## Артефакты и воспроизводимость

Каждая команда `train/evaluate` создает новую директорию `run` в `results/`:

```
results/<run_name>_<timestamp>/
├── config.yaml                       # фактическая конфигурация запуска
├── logs/                             # логи обучения/оценки
├── checkpoints/                      # DeepFM / DQN чекпоинты
├── tables/
│   ├── deepfm_history.json           # история train_static
│   ├── dqn_history.json              # история train_dqn + final_evaluation
│   ├── h1_adaptability.json          # H1 метрики
│   ├── h2_long_term.json             # H2 метрики
│   ├── h3_novelty_ablation.json      # H3 ablation
│   └── evaluation_summary.json       # сводный отчет
└── figures/                          # графики reward/coverage/novelty
```

Для OULAD команда `data analyze` генерирует таблицы и графики в
`data/processed/oulad/{tables,figures}/`.

Используются фиксированные параметры seed (см. `seed` в YAML) и сохраняется
фактический `config.yaml` в run-директории, что обеспечивает
воспроизводимость результатов.

### pytest-инварианты

Ключевые контракты системы защищены набором автоматизированных тестов:

```bash
python -m pytest tests/ -q
```

В наборе тестов проверяются:
* размерности state-векторов (ITM-Rec = 65, OULAD = 96);
* корректность action-mask (завершенные assessment, Exam в финальной фазе,
  защита от полностью нулевой маски);
* монотонность и формула `calculate_itmrec_reward` / `calculate_oulad_*`;
* `DuelingDQN.get_action(..., action_mask=...)`;
* roundtrip `DeepFMSVDPlusPlus.save_checkpoint` → `load_checkpoint`.

## Методика решения

### Датасет

Проект использует открытый образовательный датасет **ITM-Rec** ([Yong Zheng, 2023](https://github.com/irecsys/RecData/blob/main/ITM-Rec/ITM-Rec_LAK23.pdf)):

- **5,230 оценок** от 476 пользователей по 70 темам проектов
- **Многокритериальность**: каждая оценка включает 4 критерия:
  - `Rating` - общий рейтинг (1-5)
  - `App` - оценка предметной области (1-5)
  - `Data` - оценка типа данных (1-5)
  - `Ease` - оценка сложности (1-5)
- **Контекстные переменные**:
  - `Class` - специализация (DA, DM, DB)
  - `Semester` - семестр обучения
  - `Lockdown` - период COVID (PRE, DUR, POS)
- **Демографические данные**: пол, возрастная группа, семейное положение

### 1. Модель DeepFM+SVD++

Гибридная модель объединяет три компонента для прогнозирования многокритериальных рейтингов:

#### 1.1. Factorization Machine (FM)

FM моделирует попарные взаимодействия между признаками. Формула второго порядка:

$$\text{FM}^{(2)}(x) = \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

где $v_i, v_j$ - векторы эмбеддингов, $x_i, x_j$ - значения признаков.

Реализация через оптимизацию:

$$\text{FM}^{(2)}(x) = \frac{1}{2} \left[ \left( \sum_{i=1}^{n} v_i x_i \right)^2 - \sum_{i=1}^{n} (v_i x_i)^2 \right]$$

#### 1.2. Deep Neural Network (DNN)

DNN часть моделирует нелинейные взаимодействия высокого порядка:

$$\text{DNN}(x) = \sigma(W_L \cdot \text{ReLU}(W_{L-1} \cdot ... \cdot \text{ReLU}(W_1 \cdot x + b_1) ... + b_{L-1}) + b_L)$$

где $W_i$ - веса слоев, $b_i$ - смещения, $\sigma$ - функция активации (sigmoid).

#### 1.3. SVD++ для Implicit Feedback

SVD++ учитывает неявные предпочтения пользователя через историю взаимодействий:

$$r_{ui} = \mu + b_u + b_i + (p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j)^T q_i$$

где:
- $\mu$ - глобальное смещение
- $b_u, b_i$ - смещения пользователя и предмета
- $p_u, q_i$ - эмбеддинги пользователя и предмета
- $N(u)$ - множество предметов, с которыми взаимодействовал пользователь
- $y_j$ - эмбеддинги предметов из истории

#### 1.4. Объединение компонентов

Финальное предсказание для каждого критерия $c \in \{\text{Rating}, \text{App}, \text{Data}, \text{Ease}\}$:

$$\hat{r}_{ui}^{(c)} = \sigma\left( W_c^T \cdot [\text{DNN}(x), \text{FM}^{(2)}(x), \text{linear terms}] + b_c \right)$$

где $\sigma$ - sigmoid функция для нормализации к [0, 1].

### 2. Обучение с подкреплением (DQN)

#### 2.1. Формулировка задачи как MDP

- **Состояние** $s_t$: вектор размерности 65, включающий:
  - Эмбеддинг пользователя (32 dims)
  - Контекстные признаки: Class, Semester, Lockdown, time_in_semester, success_rate (10 dims)
  - Демографические признаки: пол, возрастная группа, семейное положение (6 dims)
  - История взаимодействий (15 dims)
  - Временные признаки: прогресс, длина истории (2 dims)

- **Действие** $a_t$: выбор темы проекта из набора {0, 1, ..., 69}

- **Награда** $r_t$: многокритериальная функция вознаграждения

#### 2.2. Функция вознаграждения

Вознаграждение учитывает несколько критериев с весами:

$$r_t = w_1 \cdot \text{App} + w_2 \cdot \text{Data} + w_3 \cdot \text{Ease} + w_4 \cdot \text{Novelty}$$

где базовые веса: $w_1 = 0.5$, $w_2 = 0.3$, $w_3 = 0.15$, $w_4 = 0.05$.

Веса адаптируются в зависимости от контекста:
- При Lockdown ∈ {DUR, POS}: $w_3 = 0.25$, $w_1 = 0.45$
- При Class = DB: $w_2 = 0.35$

Дополнительно учитываются:
- **Новизна**: $\text{Novelty} = 1 - \text{mean cosine similarity}(\text{последние 3 темы})$
- **Демографические модификаторы**: 
  - Если married=1: ×0.9
  - Если возраст ∈ {<20, 20-25}: ×1.1

#### 2.3. Dueling DQN архитектура

Dueling DQN разделяет оценку Q-функции на компоненты ценности состояния и преимущества действий:

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

где:
- $V(s)$ - ценность состояния (value stream)
- $A(s, a)$ - преимущество действия (advantage stream)
- $\mathcal{A}$ - пространство действий

Это позволяет агенту лучше оценивать ценность состояний независимо от конкретных действий.

#### 2.4. Обучение с использованием временных разностей (Temporal Difference Learning)

Оптимизация выполняется путем минимизации ошибки временной разности:

$$\mathcal{L} = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - d) - Q(s, a) \right)^2 \right]$$

где:
- $\gamma$ - коэффициент дисконтирования (0.99)
- $Q_{\text{target}}$ - целевая сеть (target network)
- $d$ - флаг завершения эпизода
- $\mathcal{D}$ - буфер воспроизведения опыта

#### 2.5. Приоритизированное воспроизведение опыта (Prioritized Experience Replay)

Приоритеты переходов в буфере определяются на основе величины ошибки временной разности:

$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$

где $p_i = |\delta_i| + \epsilon$ - приоритет перехода $i$, $\alpha$ - параметр важности (0.6).

Компенсация смещения при выборке из приоритизированного буфера:

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta}$$

где $\beta$ - параметр компенсации (начинается с 0.4, увеличивается до 1.0).

#### 2.6. Синхронизация целевой сети

Применяется комбинация двух стратегий обновления целевой сети:
- **Полное обновление** (каждые 100 шагов): $\theta_{\text{target}} \leftarrow \theta$
- **Частичное обновление** (остальные шаги): $\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}$

где $\tau = 0.01$ - коэффициент мягкого обновления.

#### 2.7. Стратегия исследования: ε-жадный алгоритм

$$\pi(a|s) = \begin{cases}
\text{случайное действие} & \text{с вероятностью } \epsilon \\
\arg\max_a Q(s, a) & \text{с вероятностью } 1 - \epsilon
\end{cases}$$

где $\epsilon$ уменьшается от 1.0 до 0.01 с коэффициентом затухания 0.995.

### 3. Образовательная среда моделирования

Компонент `EducationalEnvironment` реализует интерпретатор взаимодействия студента с системой:

1. **Генерация обратной связи**: использует фактические оценки из датасета или предсказания модели DeepFM+SVD++
2. **Вычисление вознаграждения**: многокритериальная функция с адаптацией к контексту
3. **Ведение истории**: накопление истории взаимодействий студента
4. **Условия завершения сеанса**: сеанс заканчивается при выполнении одного из условий:
   - Достижении максимальной длины траектории (10 шагов)
   - Низком среднем вознаграждении (< 0.3)
   - Достижении достаточного разнообразия (≥ 5 уникальных рекомендаций)

## Архитектура системы

### Описание системы

Диаграммы архитектуры реализованы в формате XML для приложения [draw.io](https://app.diagrams.net/). Процедура просмотра:

1. Откройте веб-приложение [draw.io](https://app.diagrams.net/)
2. Выберите File → Open from → Device
3. Загрузите файл из папки `schemes/`

**Доступные диаграммы:**
- `schemes/main_scheme.xml` - Общая архитектура гибридной системы
- `schemes/agent.xml` - Детальная архитектура Dueling DQN агента
- `schemes/data.xml` - Формирование вектора состояния
- `schemes/env_train.xml` - Процесс обучения и взаимодействия со средой
- `schemes/buffer.xml` - Механизм Prioritized Experience Replay

Система состоит из двух основных модулей:

1. **DeepFM+SVD++ Модуль** - предсказывает многокритериальные рейтинги на основе:
   - Данных студента (история, предпочтения)
   - Данных о контенте (характеристики тем)
   - Контекстных данных (специализация, семестр, период обучения)

2. **DQN Агент** - принимает решения о рекомендациях на основе:
   - Состояния $s_t$ (65-мерный вектор)
   - Q-значений для всех возможных действий
   - ε-жадной стратегии выбора

### Архитектура агента Dueling DQN

*Диаграмма: `schemes/agent.xml` (просмотр в [draw.io](https://app.diagrams.net/))*

Архитектура состоит из следующих компонентов:

1. **Слои признаков** (общее представление):
   - Linear(65 → 256) → BatchNorm → ReLU → Dropout(0.2)
   - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.2)
   - Linear(128 → 64) → BatchNorm → Dropout(0.2)

2. **Поток ценности** (оценка состояния):
   - Linear(64 → 32) → ReLU → Linear(32 → 1)
   - Выход: $V(s)$

3. **Поток преимуществ** (оценка действий):
   - Linear(64 → 32) → ReLU → Linear(32 → 70)
   - Выход: $A(s, a)$ для всех 70 действий

4. **Функция агрегации (Dueling)**:
   $$Q(s, a) = V(s) + A(s, a) - \frac{1}{70} \sum_{a'} A(s, a')$$

### Конструирование вектора состояния

*Диаграмма: `schemes/data.xml` (просмотр в [draw.io](https://app.diagrams.net/))*

Вектор состояния состоит из пяти компонентов:

1. **Эмбеддинг студента** (32 dims): извлекается из `model.user_emb_fm(u)`
2. **Контекст** (10 dims): OneHot кодирование Class (3) + Semester (2) + Lockdown (3) + time_in_semester (1) + success_rate (1)
3. **Демографические признаки** (6 dims): Пол (1) + Возрастная группа (4) + Семейное положение (1)
4. **История** (15 dims): последние взаимодействия, закодированные в фиксированный вектор
5. **Временные признаки** (2 dims): прогресс и длина истории

**Итого**: 32 + 10 + 6 + 15 + 2 = **65 измерений**

### Цикл обучения

*Диаграмма: `schemes/env_train.xml` (просмотр в [draw.io](https://app.diagrams.net/))*

Процесс обучения включает следующие этапы:

1. **Сбор данных из среды**:
   - Агент воспринимает состояние $s_t$
   - Выбирает действие $a_t$ используя ε-жадный алгоритм
   - Окружение возвращает результат $(r_t, s_{t+1}, \text{done})$

2. **Запись опыта**:
   - Переход $(s_t, a_t, r_t, s_{t+1}, \text{done})$ сохраняется в приоритизированный буфер

3. **Оптимизация**:
   - Выборка батча из буфера с учетом приоритета
   - Вычисление ошибки временной разности
   - Обновление приоритетов в буфере
   - Градиентный спуск для обновления весов Q-сети

### Механизм приоритизированного буфера воспроизведения

*Диаграмма: `schemes/buffer.xml` (просмотр в [draw.io](https://app.diagrams.net/))*

Приоритизированный буфер воспроизведения реализует следующий функционал:

- **Емкость**: 10,000 переходов
- **Приоритеты**: основаны на TD-ошибке
- **Выборка**: пропорциональна приоритетам с параметром $\alpha = 0.6$
- **Важность**: компенсация смещения через importance sampling с $\beta$ (0.4 → 1.0)

## Показатели качества

Система оценивается посредством следующих показателей:

### Метрики краткосрочного качества

- **Precision@K**: доля релевантных рекомендаций среди топ-K
- **Recall@K**: доля найденных релевантных предметов
- **F1@K**: гармоническое среднее Precision и Recall
- **Coverage**: доля уникальных рекомендованных предметов от общего каталога
- **Diversity**: разнообразие рекомендаций (косинусное расстояние между предметами)
- **Novelty**: средняя обратная популярность рекомендованных предметов

### Метрики долгосрочного качества

- **Совокупное дисконтированное вознаграждение (CDR)**: 
  $$\text{CDR} = \sum_{t=0}^{T} \gamma^t r_t$$
  
- **Коэффициент удержания**: доля этапов с вознаграждением выше установленного порога (0.5)

- **Тренд улучшения вознаграждения**: пространственная тенденция возрастания вознаграждения во времени (вычисляется линейной регрессией по участкам траектории)

- **Динамика расширения каталога**: количество уникальных предметов покрытия во времени

## Jupyter ноутбуки

Интерактивные сценарии строятся на `src.api` (см. также [notebooks/README.md](notebooks/README.md)).

### Быстрый старт

- **`notebooks/00_quickstart.ipynb`** — рекомендуемая точка входа: конфиг и данные (ITM-Rec или OULAD), обучение DeepFM+SVD++ и Dueling DQN, оценка гипотез H1/H2/H3, визуализация траекторий и сводные таблицы/графики в `results/`.

### Базовый датасет ITM-Rec

- **`notebooks/01_data_analysis.ipynb`** — разведочный анализ ITM-Rec: распределения по четырем критериям, влияние Class, Semester, Lockdown, корреляции и демографические профили.
- **`notebooks/02_model_development.ipynb`** — *устаревший* пошаговый разбор обучения DeepFM, среды и DQN; для обучения предпочтительны `00_quickstart` или CLI/API.
- **`notebooks/03_testing.ipynb`** — *устаревший* набор базовых прогонов и сравнений с random/popularity; для оценки предпочтительны `06_hypotheses` и `07_trajectories`.

### Расширенный сценарий OULAD

- **`notebooks/04_oulad_data.ipynb`** — EDA OULAD: структура таблиц, прокси-критерии (Mastery, Engagement, SelfRegulation, Outcome), VLE/assessment и временная динамика.
- **`notebooks/05_oulad_model.ipynb`** — полный цикл для OULAD: подготовка данных, DeepFM на недельных траекториях, DQN с action mask и долгосрочная оценка.

### Гипотезы и визуализация

- **`notebooks/06_hypotheses.ipynb`** — формальная проверка H1 (адаптивность к контексту, AdaptabilityScore), H2 (CDR, удержание, learning slope, статистика), H3 (абляция: контекст, демография, новизна) для выбранного датасета.
- **`notebooks/07_trajectories.ipynb`** — качественный анализ: накопленные награды, сравнение DQN с baseline (random, popularity, static), переходы, охват и новизна.

### Типовые сценарии (время ориентировочно)

| Цель | Ноутбуки |
|------|----------|
| Минимум: один полный прогон | `00_quickstart` |
| Глубина по гипотезам | `00_quickstart` → `06_hypotheses` |
| Фокус на OULAD | `04_oulad_data` → `00_quickstart` (конфиг OULAD) → `06_hypotheses` |
| Качество траекторий | `07_trajectories` (после обучения) |

## Необходимые зависимости

Основные библиотеки и модули:
- `torch>=2.0.0` - Фреймворк для глубокого обучения
- `pandas>=2.0.0`, `numpy>=1.24.0` - Инструменты обработки данных
- `scikit-learn>=1.3.0` - Утилиты машинного обучения и вычисления метрик
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Библиотеки для визуализации
- `kagglehub>=0.2.0` - Загрузка датасетов из Kaggle
- `tqdm>=4.65.0` - Индикаторы прогресса
- `scipy>=1.10.0` - Статистические функции и алгоритмы

Полный список в `requirements.txt`.

## Правовая информация

Проект распространяется в соответствии с лицензией MIT - см. файл [LICENSE](LICENSE) для полной информации.

## Разработчик

Анастасия Золотарева