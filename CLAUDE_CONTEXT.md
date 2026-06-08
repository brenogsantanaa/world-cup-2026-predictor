# Claude Context: Sports Prediction Engine

This file gives Claude or another AI coding assistant enough context to work efficiently on this project without losing the learning goals. The user wants to build this project step by step, with mentorship, not have an assistant generate everything at once.

## Project Name

Sports Prediction Engine

## High-Level Goal

Build a beginner-friendly but resume-worthy end-to-end machine learning project for sports outcome prediction.

The first training ground is NBA game prediction. Later, the same architecture should be adapted to soccer / FIFA World Cup match prediction and tournament simulation.

The important idea is not just to train one model. The project should become a reusable sports modeling system where:

- Sport-specific code handles data loading and feature engineering.
- Shared core code handles model training, evaluation, prediction, persistence, and eventually simulation.
- Notebooks are used for learning and exploration.
- Stable logic gradually moves from notebooks into Python modules.

## User Profile

The user:

- Knows programming and software development fundamentals.
- Is a beginner in machine learning.
- Wants to understand every design choice.
- Wants a build-first, mentor-style workflow.
- Wants to be challenged when a design choice is poor.
- Wants the project to be resume-worthy.
- Does not want large unexplained code dumps.

Teaching style should be:

1. Define new concepts simply.
2. Explain why the concept matters.
3. Show a small example.
4. Apply it to this project.
5. Point out common beginner mistakes.
6. Move in small steps.

## Current Repository State

The repo is currently an early scaffold.

Existing files and folders:

```text
sports-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── CLAUDE_CONTEXT.md
├── app/
│   └── .gitkeep
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_nba_data_exploration.ipynb
├── src/
│   └── sports_predictor/
│       ├── __init__.py
│       ├── core/
│       │   └── __init__.py
│       ├── nba/
│       │   └── __init__.py
│       └── soccer/
│           └── __init__.py
└── tests/
    └── .gitkeep
```

The local virtual environment `.venv/` may exist but is ignored and should not be committed.

## Current README Summary

The README frames the project as a guided beginner-friendly ML project.

The first goal is:

- Predict NBA game outcomes.
- Later adapt the same project structure to soccer and FIFA World Cup simulation.

The README's first milestone is intentionally not model training. It is:

- Load NBA game data.
- Inspect the dataset.
- Understand its columns.
- Create a clean target label:

```text
home_team_win = 1 if the home team won
home_team_win = 0 if the home team lost
```

That label is the value the first model will learn to predict.

## Current Dependencies

`requirements.txt` currently contains:

```text
pandas
numpy
scikit-learn
xgboost
jupyter
matplotlib
seaborn
pytest
```

The user also decided to use the NBA API approach instead of manually downloading CSVs.

Recommended next dependency to add when implementing ingestion:

```text
nba_api
```

Do not add many new dependencies early. Keep the stack beginner-friendly.

## Environment Notes

The environment was upgraded from macOS system Python 3.9 to a Python 3.12 virtual environment.

Verified environment at one point:

```text
Python 3.12.13
pandas 3.0.3
numpy 2.4.5
sklearn 1.8.0
xgboost 3.2.0
```

XGBoost initially failed on macOS because `libomp` was missing. It was fixed by installing:

```bash
brew install libomp
```

Important teaching note: this was a good example that many ML libraries rely on compiled native code, not only pure Python.

## Data Source Decision

The user prefers using an API because it feels more realistic.

Recommended first NBA data source:

```text
nba_api
```

Reasoning:

- No API key required.
- Wraps NBA.com stats.
- Returns pandas-friendly DataFrames.
- Better for learning realistic sports data ingestion.
- Richer than many simple free REST APIs.

Tradeoff:

- It is an unofficial wrapper around NBA.com and can be slow or flaky.
- We should save raw API responses locally so notebooks are reproducible and do not repeatedly hit the API.

Suggested ingestion flow:

```text
NBA API -> raw pandas DataFrame -> saved CSV in data/raw/ -> notebook exploration -> cleaned data in data/processed/
```

## Important Workflow Rule

The user explicitly said:

> I want to build together with you, step by step, so I can learn, and not just have you do everything.

Therefore, do not jump straight into implementing the whole system.

Preferred collaboration rhythm:

1. Explain the next small goal.
2. Explain why it matters.
3. Ask the user to run or review a small command/code cell when useful.
4. Inspect results together.
5. Only then write or refactor code.

For code edits:

- Keep changes small.
- Explain the purpose before or after the edit.
- Avoid generating large files without walking the user through them.
- Prefer notebooks for first exposure to a concept.
- Move code into `src/` only after it has been explored and understood.

## Machine Learning Roadmap

### Phase 1: Environment Setup

Goal: make the project runnable and reproducible.

Tasks:

- Use a virtual environment.
- Install dependencies.
- Verify imports.
- Explain why each tool exists.
- Eventually add `pyproject.toml` for editable installs with `pip install -e .`.

Concepts to teach:

- Virtual environments.
- Dependency management.
- Why ML libraries often rely on native compiled code.
- Why reproducibility matters.

### Phase 2: NBA Data Pipeline

Goal: collect historical NBA game data and save it locally.

Target data fields:

- Game date.
- Season.
- Home team.
- Away team.
- Home score.
- Away score.
- Win/loss result.

Likely starting API:

- `nba_api`.

Suggested implementation approach:

- First use a notebook cell to fetch a small sample of NBA games.
- Inspect the returned DataFrame.
- Save the raw result to `data/raw/`.
- Later move the API fetching code into `src/sports_predictor/nba/`.

Concepts to teach:

- API ingestion.
- Raw vs processed data.
- DataFrame structure.
- Rows and columns.
- Why we cache raw data locally.

### Phase 3: Exploratory Data Analysis

Goal: understand the dataset before modeling.

Current notebook:

```text
notebooks/01_nba_data_exploration.ipynb
```

It currently introduces:

- EDA.
- DataFrames.
- `head()`.
- `shape`.
- `columns`.
- `info()`.
- `describe()`.

The notebook still assumes a placeholder CSV called `games.csv`. This should eventually be updated once API ingestion produces a real raw file.

Concepts to teach:

- What one row represents.
- What each important column means.
- Missing values.
- Data types.
- Basic summary statistics.
- Why inspecting data comes before training.

### Phase 4: Target Label

Goal: create the first supervised learning label.

For NBA:

```text
home_team_win = 1 if home team score > away team score
home_team_win = 0 otherwise
```

Concepts to teach:

- Supervised learning.
- Features `X`.
- Label/target `y`.
- Binary classification.
- Why labels must be known during training but not used as input features.

Beginner mistake to avoid:

- Accidentally including final score columns as model features when predicting a pre-game outcome. That is data leakage.

### Phase 5: Feature Engineering

Goal: create pre-game features that would have been known before each game started.

Candidate NBA features:

- Team historical win percentage.
- Last 5 games win rate.
- Average points scored before this game.
- Average points allowed before this game.
- Home court advantage.
- Rest days.
- Head-to-head history.

Critical rule:

- Features for a game must only use data from before that game.

Concepts to teach:

- Feature engineering.
- Rolling windows.
- Grouping by team.
- Chronological calculations.
- Leakage.
- Why feature quality often matters more than model complexity.

### Phase 6: First ML Model

Goal: train a simple baseline model.

Start with:

```text
Logistic regression
```

Prediction target:

```text
Home team wins = 1
Home team loses = 0
```

Teach:

- `X` as the feature matrix.
- `y` as the label vector.
- Train/test split.
- Why chronological split is better than random split for sports time series.
- Model fitting.
- Prediction probabilities.

Metrics:

- Accuracy.
- Precision.
- Recall.
- Log loss.

Important:

- Log loss is especially useful because sports prediction cares about probability quality, not only right/wrong picks.

### Phase 7: Better Models

Goal: compare models after the baseline works.

Models:

- Random forest.
- Gradient boosting.
- XGBoost.

Teach:

- Why a baseline matters.
- Overfitting.
- Model complexity.
- Feature importance.
- Why better accuracy is not always better probability calibration.

### Phase 8: Reusable Architecture

Goal: refactor stable logic into reusable modules.

Desired function names:

```python
load_data()
engineer_features()
train_model()
evaluate_model()
predict_game()
```

Recommended module boundaries:

```text
src/sports_predictor/core/
    Sport-agnostic utilities:
    - train/test splitting
    - model training wrappers
    - evaluation metrics
    - saving/loading models
    - prediction probability helpers

src/sports_predictor/nba/
    NBA-specific logic:
    - NBA API ingestion
    - NBA raw data normalization
    - NBA feature engineering
    - NBA team identifiers

src/sports_predictor/soccer/
    Soccer-specific logic:
    - soccer data ingestion
    - soccer feature engineering
    - Elo or team rating features
    - tournament-stage features
```

Architecture principle:

```text
Sport-specific code creates a standard modeling table.
Core code trains and evaluates models from that table.
```

### Phase 9: Soccer / FIFA Transfer

Goal: adapt the same pipeline to soccer after the NBA version works.

Potential soccer features:

- Elo ratings.
- Goals scored.
- Goals conceded.
- Last 5 matches.
- Neutral venue.
- Tournament stage.

Teach:

- What is reusable.
- What is sport-specific.
- Why different sports require different features.
- How the same ML pipeline can support multiple domains.

### Phase 10: Tournament Simulation

Goal: use match probabilities to simulate tournaments.

Method:

- Generate match win probabilities.
- Run many Monte Carlo simulations.
- Count outcomes.

Outputs:

- Group advancement probabilities.
- Quarterfinal probabilities.
- Semifinal probabilities.
- Final probabilities.
- Champion probabilities.

Teach:

- Monte Carlo simulation.
- Probability distributions.
- Calibration.
- Why a model that predicts probabilities can power simulations.

### Phase 11: Betting Odds Validation

Goal: compare model probabilities with sportsbook implied probabilities for experimentation only.

This is not financial advice.

Track:

- Model predicted probability.
- Bookmaker implied probability.
- Difference / edge.
- Expected value.

Teach:

- Implied probability.
- Vig / bookmaker margin.
- Expected value.
- Why validation against markets is interesting but risky.

## Recommended Near-Term Next Steps

The next best small step is not to build the whole model. It is:

1. Add `nba_api` to the project dependencies.
2. Verify it imports in the `.venv`.
3. In a notebook or small scratch cell, fetch a small amount of NBA schedule/game data.
4. Inspect the returned DataFrame.
5. Save a raw CSV into `data/raw/`.
6. Update `01_nba_data_exploration.ipynb` to load that raw CSV.
7. Create `home_team_win`.

Only after the user understands that flow should code be moved into `src/sports_predictor/nba/`.

## Suggested File Evolution

Near-term additions:

```text
src/sports_predictor/nba/api.py
src/sports_predictor/nba/cleaning.py
src/sports_predictor/nba/features.py
src/sports_predictor/core/splitting.py
src/sports_predictor/core/training.py
src/sports_predictor/core/evaluation.py
tests/test_nba_features.py
```

Do not create all of these at once. They are a destination, not the immediate next step.

## Coding Style Guidance

Use beginner-readable code.

Prefer:

- Clear function names.
- Small functions.
- Explicit intermediate variables.
- Comments only where they explain non-obvious reasoning.
- Type hints once functions stabilize.

Avoid:

- Clever one-liners.
- Premature abstractions.
- Large framework decisions early.
- Hidden magic.
- Refactoring before the user understands the notebook version.

## Notebook Guidance

Notebooks are for:

- Learning.
- Exploration.
- Seeing outputs.
- Asking questions about the data.

Modules are for:

- Reusable logic.
- Tested code.
- Functions that should run the same way every time.

Recommended pattern:

```text
Notebook first -> understand -> extract stable code to src/ -> test it -> reuse in notebook
```

## Testing Philosophy

The first tests should protect logic that is easy to get subtly wrong:

- `home_team_win` label creation.
- Rolling features that must not include the current game.
- Chronological train/test split.
- Evaluation metric wrappers.

Tests should use tiny fake DataFrames at first so the user can understand them.

## Resume Positioning

Possible resume summary:

```text
Sports Prediction Engine: End-to-end Python machine learning project for predicting NBA game outcomes, using pandas, scikit-learn, XGBoost, and Jupyter. Designed a reusable sports modeling pipeline with API-based data ingestion, feature engineering, model evaluation, and future support for soccer tournament simulation.
```

Possible resume bullets:

- Designed a reusable sports prediction pipeline to train and evaluate machine learning models on historical NBA game data.
- Built a structured Python project with separate modules for raw data, feature engineering, model training, evaluation, notebooks, and future app deployment.
- Engineered predictive features such as team win percentage, recent form, scoring averages, home-court advantage, rest days, and head-to-head history while avoiding data leakage.
- Trained baseline and advanced classification models, including logistic regression, random forests, gradient boosting, and XGBoost, to predict home-team win probabilities.
- Planned architecture for adapting the same modeling pipeline to FIFA World Cup match prediction and Monte Carlo tournament simulations.

## Coordination Notes for Claude

If Claude is working alongside Cursor/GPT:

- Keep changes small and explain them.
- Do not overwrite user work.
- Check existing files before editing.
- Preserve the mentor workflow.
- Prefer asking the user to inspect outputs rather than silently doing everything.
- If adding code, mention what concept the code teaches.
- If making architecture decisions, explain the tradeoff.
- Avoid jumping ahead to Streamlit, betting odds, or tournament simulation before the NBA baseline is working.

The main success criterion is not just a working project. The main success criterion is that the user understands how to build future ML systems independently.
