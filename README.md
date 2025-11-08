# Diet Planner ML

A small machine-learning-based diet planner and meal recommender project.

This repository contains scripts to train a calorie (or nutrition) prediction model and a simple recommender that suggests meals based on previous meal history. It's a student project for experimentation and learning.

## Contents

- `diet_planner_ml.py` - Main script / demo (entry point) for the diet planner (may provide a combined interface to the model and recommender).
- `recommender.py` - Recommender system that suggests meals from historical data.
- `train_calorie_model.py` - Script to train the calorie (or nutrition) prediction model.
- `meal_history.csv` - Example dataset with past meals used for training and recommendations.

## Quick overview

- The project reads meal history from `meal_history.csv`.
- `train_calorie_model.py` is used to train a model (e.g., linear regression or other) to predict calories or other nutrition values.
- `recommender.py` uses historical data and model outputs to recommend meals.
- `diet_planner_ml.py` ties components together to provide a simple planner/CLI usage (see below).

## Requirements

The project uses Python 3.8+ and common ML/data libraries. If you don't have a `requirements.txt`, install the typical packages below:

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib
```

If you prefer, create a `requirements.txt` with the packages above and run:

```powershell
pip install -r requirements.txt
```

## Usage (PowerShell)

1. Train the model (if training script exists and is configured):

```powershell
python train_calorie_model.py
```

2. Run the recommender to get meal suggestions:

```powershell
python recommender.py
```

3. Run the main planner/demo:

```powershell
python diet_planner_ml.py
```

Notes:
- The exact command-line options depend on how the scripts are implemented. Open the top of each script to see available CLI flags or expected input files.
- If your scripts save or load models (e.g., with `joblib`), ensure the paths used in the scripts match files in the repository.

## Data

`meal_history.csv` should contain historical meal entries. Typical columns you might expect:
- date, meal_name, calories, protein, carbs, fat, user_notes, etc.

If your CSV has a different schema, update the scripts to match the column names.

## How it works (short)

- Data is loaded from `meal_history.csv` and preprocessed.
- A model (from `train_calorie_model.py`) is trained to predict calories or other nutrition info.
- `recommender.py` uses simple heuristics or model outputs to suggest meals similar to user preferences or nutritional needs.

## Development & Contribution

This repository is a student project. If you'd like to contribute:
- Fork or clone the repo
- Create a branch for your feature
- Add tests where appropriate
- Open a pull request describing your change

## Potential improvements / Next steps

- Add a `requirements.txt` or a `pyproject.toml` for reproducible installs.
- Add a small unit test suite for preprocessing and the recommender logic.
- Add example input/output and a small demo notebook.
- Improve the recommender using collaborative filtering or embeddings.

## License

You can add a license file (e.g., MIT) if you want to make the repo public. Currently no license file is included — check with the project owner.

## Contact

Project owner / author: Jyoti

---

If you'd like, I can also:
- generate a `requirements.txt` with pinned versions,
- add short usage examples inside each script,
- or create a CONTRIBUTING.md and LICENSE file. Tell me which next and I'll implement it.
