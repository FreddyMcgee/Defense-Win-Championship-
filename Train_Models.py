import pandas as pd
from numpy.random.bit_generator import DECIMAL_RE

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)

from Modeling import train_data, off_features

df = pd.read_parquet("team_season_data.parquet")

off_features = [
    "points_scored_per_game",
    "yards_per_play",
    "passing_yards_per_game",
    "rushing_yards_per_game",
    "tds_per_game",
    "completion_pct",
    "passing_epa_per_game",
    "rushing_epa_per_game",
    "turnovers_per_game",
    "sacks_allowed_per_game"
]

#
#Training: 2015 - 2021
#
# Training: 2015 - 2021
train_data = df[df["season"] <= 2021]

validation_data = df[
    df["season"].between(2022, 2023)

]

X_train = train_data[off_features]
X_test = validation_data[off_features]

y_train = train_data["playoff_class"]
y_val = validation_data["playoff_class"]

modeldt = DecisionTreeClassifier(max_depth=4,
                                class_weight="balanced",
                                random_state=42 )
# Fit Model
modeldt.fit(X_train, y_train)

# Predict
y_pred = modeldt.predict(X_test)

# Evaluation

print("Decision Tree - Offense Only")

print("Training observations:", len(y_train))
print("Validation observations:", len(y_val))

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_val, y_pred)
)

print(
    "Macro F1:",
    f1_score(
        y_val,
        y_pred,
        average="macro",
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        y_pred,
        labels=[1, 2, 3, 4, 5]
    )
)



# Defensive Stats

def_features = [
    "points_allowed_per_game",
    "yards_allowed_per_play",
    "passing_yards_allowed_per_game",
    "rushing_yards_allowed_per_game",
    "passing_tds_allowed_per_game",
    "rushing_tds_allowed_per_game",
    "passing_epa_allowed_per_game",
    "rushing_epa_allowed_per_game",
    "def_sacks_per_game",
    "def_interceptions_per_game"
]


# DEFENSIVE DECISION TREE


# Defensive training and validation features
X_def_train = train_data[def_features]
X_def_val = validation_data[def_features]

# Create Decision Tree
modeldt_defense = DecisionTreeClassifier(
    max_depth=4,
    class_weight="balanced",
    random_state=42
)

# Train model
modeldt_defense.fit(X_def_train, y_train)

# Make predictions
y_pred_defense = modeldt_defense.predict(X_def_val)

# Evaluate model
print("\nDecision Tree - Defense Only")

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_val, y_pred_defense)
)

print(
    "Macro F1:",
    f1_score(
        y_val,
        y_pred_defense,
        average="macro",
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        y_pred_defense,
        labels=[1, 2, 3, 4, 5]
    )
)

