import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Load the clean dataset

full_season = pl.read_parquet("team_season_data.parquet")

#Verify the data loaded correctly
# print("Dataset shape: ", full_season.shape)
# print(full_season.head())

# Offensive Features
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
    "sacks_allowed_per_game",

]


X_offense = full_season.select(off_features)

# DEFENSIVE FEATURES
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

X_defense = full_season.select(def_features)



# COMBINED OFFENSE + DEFENSE
X_combined = full_season.select(
    off_features + def_features
)

# TARGET VARIABLE
y = full_season["playoff_class"]

# VERIFIED DIMENSIONS
print("Offense:", X_offense.shape)
print("Defense:", X_defense.shape)
print("Combined:", X_combined.shape)
print("Target:", y.shape)




# Split NFL seasons into training and testing
all_seasons = sorted(full_season["season"].unique().to_list())

train_seasons, test_seasons = train_test_split(
    all_seasons,
    test_size=2,
    shuffle=False
)

train_data = full_season.filter(
    pl.col("season").is_in(train_seasons)
)

test_data = full_season.filter(
    pl.col("season").is_in(test_seasons)
)

print("\nTraining Seasons:", train_seasons)
print("Testing Seasons:", test_seasons)

print("Training Dataset:", train_data.shape)
print("Testing Dataset:", test_data.shape)

# ==================================================
# STEP 6: SEPARATE X_train, X_test, y_train, y_test
# ==================================================

# Offensive training and testing features
X_offense_train = train_data.select(off_features)
X_offense_test = test_data.select(off_features)

# Defensive training and testing features
X_defense_train = train_data.select(def_features)
X_defense_test = test_data.select(def_features)

# Combined training and testing features
X_combined_train = train_data.select(
    off_features + def_features
)

X_combined_test = test_data.select(
    off_features + def_features
)

# Target variables
y_train = train_data["playoff_class"]
y_test = test_data["playoff_class"]



# STEP 7: VERIFY TRAINING AND TESTING SHAPES


print("\nTraining and Testing Shapes:")

print("Offense Train:", X_offense_train.shape)
print("Offense Test:", X_offense_test.shape)

print("Defense Train:", X_defense_train.shape)
print("Defense Test:", X_defense_test.shape)

print("Combined Train:", X_combined_train.shape)
print("Combined Test:", X_combined_test.shape)


# ==================================================
# STEP 8: FEATURE SCALING
# ==================================================

# Create separate scalers
offense_scaler = StandardScaler()
defense_scaler = StandardScaler()
combined_scaler = StandardScaler()


# OFFENSE
X_offense_train_scaled = offense_scaler.fit_transform(
    X_offense_train.to_numpy()
)

X_offense_test_scaled = offense_scaler.transform(
    X_offense_test.to_numpy()
)


# DEFENSE
X_defense_train_scaled = defense_scaler.fit_transform(
    X_defense_train.to_numpy()
)

X_defense_test_scaled = defense_scaler.transform(
    X_defense_test.to_numpy()
)


# COMBINED
X_combined_train_scaled = combined_scaler.fit_transform(
    X_combined_train.to_numpy()
)

X_combined_test_scaled = combined_scaler.transform(
    X_combined_test.to_numpy()
)


# Verify scaling
print("\nScaled Dataset Shapes:")

print("Offense Train:", X_offense_train_scaled.shape)
print("Offense Test:", X_offense_test_scaled.shape)

print("Defense Train:", X_defense_train_scaled.shape)
print("Defense Test:", X_defense_test_scaled.shape)

print("Combined Train:", X_combined_train_scaled.shape)
print("Combined Test:", X_combined_test_scaled.shape)

print("\nOffense training mean:",
      X_offense_train_scaled.mean())

print("Offense training standard deviation:",
      X_offense_train_scaled.std())