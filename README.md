# Defense Wins Championships?
### Predicting NFL Playoff Success Using Offensive and Defensive Performance

## Project Overview

This data mining project investigates whether offensive or defensive performance is a stronger predictor of NFL postseason success.

Using NFL team statistics from the **2015–2025 seasons**, we will compare machine learning models trained on offensive statistics, defensive statistics, and a combination of both.

## Research Question

**Does defense or offense better predict how far an NFL team advances in the playoffs?**

## Dataset

- **Source:** NFL data accessed using `nflreadpy`
- **Seasons:** 2015–2025
- **Observations:** 352 team-seasons
- **Final dataset:** 59 columns, including the target variable
- **Target:** `playoff_class`

## Postseason Classification

| Class | Description |
|---|---|
| 1 | Missed the playoffs |
| 2 | Eliminated in Wild Card or Divisional Round |
| 3 | Lost the conference championship |
| 4 | Lost the Super Bowl |
| 5 | Won the Super Bowl |

## Project Progress

### Completed
- Collected NFL team and schedule data.
- Filtered regular-season statistics.
- Aggregated offensive statistics by team and season.
- Calculated defensive performance metrics.
- Combined offensive and defensive datasets.
- Created postseason classification labels.
- Validated the dataset for missing values, duplicate team-seasons, NaN, and infinity.

### In Progress
- Selecting offensive and defensive modeling features.
- Preparing training and testing datasets.
- Defining machine learning methods and evaluation procedures.

### Planned
- Decision Tree classification.
- Random Forest classification.
- K-Nearest Neighbors classification.
- Compare offense-only, defense-only, and combined models.
- Evaluate predictions using balanced accuracy, macro F1-score, and confusion matrices.
- Analyze which statistics best predict postseason success.

## Technologies

Python, Polars, nflreadpy, Git, GitHub, and PyCharm.

## Project Status

**Current phase:** Modeling Data Preparation

This is an ongoing academic data mining project. Model results and conclusions have not yet been established.
