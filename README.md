# Defense Wins Championships?
### Predicting NFL Playoff Success Using Offensive and Defensive Performance

## Project Overview

This academic data mining project investigates whether **offensive or defensive regular-season performance is more predictive of NFL postseason success**. We use NFL team statistics from the **2015–2025 seasons** to compare offense-only, defense-only, and combined predictive feature sets.

**Research question:** Does offense or defense better predict how far an NFL team advances in the playoffs?

> **Status (October 20, 2026):** Data preprocessing and feature preparation are complete. Initial offense-only and defense-only Decision Tree models have been validated. Combined Decision Tree, Random Forest, KNN, and final holdout testing are still in progress.

## Dataset and Preprocessing

- **Source:** NFL team statistics and schedules loaded using `nflreadpy`.
- **Coverage:** 2015–2025 regular seasons, with postseason schedules used to create outcome labels.
- **Unit of analysis:** One record per team per season (**352 team-seasons**).
- **Processed data:** **61 columns**, including the postseason outcome label and two additional defensive per-game features.
- **Target variable:** `playoff_class` (five classes).
- **Quality checks:** 0 duplicate team-seasons, 0 missing values, and 0 rows containing NaN or infinity in the processed dataset.

The pipeline aggregates weekly team statistics into season-level offensive and defensive measures; joins team and opponent performance and game scores; harmonizes historical team abbreviations; and labels each team's deepest postseason achievement.

**Data interpretation note:** Points scored and allowed use final game scores, which can include defensive and special-teams scoring. Therefore, they are not exclusively offense-generated or defense-generated points.

### Postseason Classification

| Class | Team's postseason result | Team-seasons |
|---|---|---:|
| 1 | Missed the playoffs | 208 |
| 2 | Eliminated in the Wild Card or Divisional Round | 100 |
| 3 | Lost the conference championship | 22 |
| 4 | Lost the Super Bowl | 11 |
| 5 | Won the Super Bowl | 11 |
| **Total** | | **352** |

The class distribution is uneven, particularly for Super Bowl participants. We therefore use balanced accuracy and macro F1 alongside confusion matrices rather than relying only on ordinary accuracy.

## Modeling Data

We selected three candidate feature sets for controlled comparisons:

| Feature set | Number of predictors | Examples |
|---|---:|---|
| Offense only | 10 | Points scored per game, yards per play, passing/rushing yards per game, EPA, turnovers |
| Defense only | 10 | Points allowed per game, yards allowed per play, passing/rushing yards allowed, EPA allowed, sacks, interceptions |
| Combined | 20 | All 10 offensive and 10 defensive predictors |

### Season-Based Evaluation Design

We keep the seasons in chronological order to avoid using future seasons to train models evaluated on earlier ones:

| Partition | Seasons | Team-seasons | Use |
|---|---|---:|---|
| Initial model training | 2015–2021 | 224 | Fit initial models |
| Validation | 2022–2023 | 64 | Compare model approaches and settings |
| Final holdout test | 2024–2025 | 64 | Evaluate selected models only after development |

`Modeling.py` also prepares a broader **2015–2023 training / 2024–2025 testing** split (288 / 64 observations) and demonstrates `StandardScaler` fitted on the 2015–2023 portion. For KNN tuning against 2022–2023, scaling will instead be fitted **only on 2015–2021** within the validation experiment to avoid data leakage. Decision Trees do not require scaling.

## Preliminary Results — Decision Tree

Initial `DecisionTreeClassifier` experiments use `max_depth=4`, `class_weight="balanced"`, and `random_state=42` with 2015–2021 training and 2022–2023 validation data.

| Feature set | Balanced accuracy | Macro F1 |
|---|---:|---:|
| **Offense only** | **33.44%** | **0.301** |
| Defense only | 19.33% | 0.183 |
| Combined | Not yet evaluated | Not yet evaluated |

**Preliminary observation:** The offense-only Decision Tree performed better than the defense-only Decision Tree on these two validation seasons. Neither model correctly identified the two Super Bowl champions in the validation set. These early results **do not establish** that offense is generally more important than defense: the sample is small, classes are imbalanced, and additional algorithms and final holdout evaluation remain.

## Repository Files

| File | Purpose |
|---|---|
| `Cleaned Data.py` | Download and preprocess NFL statistics, create postseason labels, validate the data, and export the Parquet file |
| `Modeling.py` | Select 10 offensive and 10 defensive features, split seasons, and demonstrate scaling |
| `Train_Models.py` | Train and evaluate the current offense-only and defense-only Decision Tree classifiers |
| `team_season_data.parquet` | Locally generated processed dataset, created by `Cleaned Data.py` (not required in version control) |

### Running the Project

Install the project dependencies into your Python environment:

```bash
python -m pip install nflreadpy polars pandas numpy scikit-learn pyarrow
```

Run the files from the repository root, in order:

```bash
python "Cleaned Data.py"
python Modeling.py
python Train_Models.py
```

The preprocessing script creates `team_season_data.parquet` for the subsequent scripts. NFL data retrieval may require an internet connection. The training script currently imports `Modeling.py`, so its setup/verification output will also appear when running `Train_Models.py`.

## Progress

**Completed**
- [x] Retrieve 2015–2025 NFL team and schedule data
- [x] Build season-level offensive and defensive statistics
- [x] Join team-season data and assign five postseason outcome labels
- [x] Check for missing values, duplicates, NaN, and infinity
- [x] Select offense-only, defense-only, and combined feature sets
- [x] Establish chronological validation and test periods
- [x] Train and evaluate offense-only and defense-only Decision Trees

**Next steps**
- [ ] Train and evaluate the combined Decision Tree
- [ ] Implement Random Forest and K-Nearest Neighbors
- [ ] Tune models with training-season-only preprocessing
- [ ] Compare feature groups across algorithms
- [ ] Perform final 2024–2025 holdout evaluation
- [ ] Interpret results and complete the literature review and project documentation

## Tools

Python · `nflreadpy` · Polars · Pandas · Scikit-learn · PyCharm · Git · GitHub

*This project is ongoing. All performance figures above are preliminary validation results, not final test-set performance.*
