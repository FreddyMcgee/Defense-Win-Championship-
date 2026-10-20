import numpy as np
import pandas as pd
import nflreadpy as nfl
import polars as pl

seasons = list(range (2015,2026))

teams_df =  nfl.load_team_stats(seasons)
# Getting Game Scores
schedules = nfl.load_schedules(seasons)

reg_schedules = schedules.filter(
    pl.col("game_type") == "REG"
)



# Normalize historical team abbreviations
team_names = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS"
}

team_scores = pl.concat([

    # Home team statistics
    reg_schedules.select([
        "game_id",
        pl.col("home_team").replace(team_names).alias("team"),
        pl.col("home_score").alias("points_scored"),
        pl.col("away_score").alias("points_allowed")
    ]),

    # Away team statistics
    reg_schedules.select([
        "game_id",
        pl.col("away_team").replace(team_names).alias("team"),
        pl.col("away_score").alias("points_scored"),
        pl.col("home_score").alias("points_allowed")
    ])

])


# Finding the of the df
#  print(teams_df.shape)

# Observing the first 5 rows
# print(teams_df.head(5))

# Viewing the Columns of the df
# print(teams_df.columns)
# Results: ['season', 'week', 'team', 'season_type', 'game_id',
# 'opponent_team', 'completions', 'attempts', 'passing_yards',
# 'passing_tds', 'passing_interceptions', 'sacks_suffered',
# 'sack_yards_lost', 'sack_fumbles', 'sack_fumbles_lost',
# 'passing_air_yards', 'passing_yards_after_catch',
# 'passing_first_downs', 'passing_epa', 'passing_cpoe',
# 'passing_2pt_conversions', 'passing_10', 'passing_16',
# 'passing_20', 'passing_40', 'carries', 'rushing_yards',
# 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
# 'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
# , 'rushing_12', 'rushing_20', 'rushing_40', 'receptions', 'targets',
# 'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
# 'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs',
# 'receiving_epa', 'receiving_2pt_conversions', 'receiving_10', 'receiving_16',
# 'receiving_20', 'receiving_40', 'special_teams_tds', 'def_tackles_solo',
# 'def_tackles_with_assist', 'def_tackle_assists', 'def_tackles_for_loss',
# 'def_tackles_for_loss_yards', 'def_fumbles_forced', 'def_sacks',
# 'def_sack_yards', 'def_qb_hits', 'def_interceptions',
# 'def_interception_yards', 'def_pass_defended', 'def_tds',
# 'def_fumbles', 'def_safeties', 'def_punt_blocks', 'def_pat_blocks',
# 'def_fg_blocks', 'def_2pt_atts', 'def_2pt_made', 'misc_yards',
# 'fumble_recovery_own', 'fumble_recovery_yards_own',
# 'fumble_recovery_opp', 'fumble_recovery_yards_opp',
# 'fumble_recovery_tds', 'penalties', 'penalty_yards',
# 'timeouts', 'fumbles_forced_by_opp', 'fumbles_not_forced',
# 'fumbles_out_of_bounds', 'fumbles_total', 'fumbles_lost_total',
# 'punt_returns', 'punt_return_yards', 'kickoff_returns',
# 'kickoff_return_yards', 'fg_made', 'fg_att', 'fg_missed',
# 'fg_blocked', 'fg_long', 'fg_pct', 'fg_made_0_19', 'fg_made_20_29',
# 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_',
# 'fg_missed_0_19', 'fg_missed_20_29', 'fg_missed_30_39',
# 'fg_missed_40_49', 'fg_missed_50_59', 'fg_missed_60_',
# 'fg_made_list', 'fg_missed_list', 'fg_blocked_list',
# 'fg_made_distance', 'fg_missed_distance', 'fg_blocked_distance',
# 'pat_made', 'pat_att', 'pat_missed', 'pat_blocked', 'pat_pct',
# 'gwfg_made', 'gwfg_att', 'gwfg_missed', 'gwfg_blocked', 'gwfg_distance',
# 'pt_att', 'pt_blocked', 'pt_long', 'pt_yards', 'pt_inside_20',
# 'pt_out_of_bounds', 'pt_downed', 'pt_touchback', 'pt_fair_caught',
# 'pt_returned', 'pt_return_yards', 'pt_return_tds', 'pt_net_yards']

# Output => "REG" | "POST"
#print(teams_df['season_type'].unique())

reg_team = teams_df.filter(
    (pl.col("season_type") == "REG") &
    (
        pl.col("week") <=
        pl.when(pl.col("season") <= 2020)
        .then(17)
        .otherwise(18)
    )
)


post_team = teams_df.filter(pl.col("season_type") == "POST")
# print(post_team)


post_team = post_team.select(["season","week", "team","opponent_team","game_id"])


# print(post_team)

# Check for Duplicates
# duplicates = (
#     reg_team
    # .group_by("season", "team", "game_id")
    # .len()
    # .filter(pl.col("len") > 1)
# )
# print(duplicates)
# No Duplicates

missing_reg_values = (
    reg_team.null_count()
    .transpose(
        include_header=True,
        header_name="column",
        column_names=["nan_count"]
    )
    .filter(pl.col("nan_count") > 0)
    .sort("nan_count",descending=True)
)

# print(missing_reg_values.shape)
# print(missing_reg_values)
#Missing Values for FGs statistics because some games don't include FGs

reg_off_weekly = reg_team.select([
    "season",
    "week",
    "team",
    "game_id",
    "opponent_team",
    "passing_yards",
    "completions",
    "attempts",
    "sacks_suffered",
    "sack_yards_lost",
    "passing_tds",
    "passing_first_downs",
    "passing_interceptions",
    "passing_epa",
    "rushing_yards",
    "rushing_first_downs",
    "carries",
    "rushing_tds",
    "rushing_epa",
    "fumbles_lost_total"

])



reg_off_season = (
    reg_off_weekly
    .group_by(["season", "team"])
    .agg([
        pl.len().alias("games_played"),
        # Adding Total YARDS
        (pl.col("passing_yards").sum() +
         pl.col("rushing_yards").sum() +
         pl.col("sack_yards_lost").sum()).alias("total_yards"),

        # ADDING TOTAL TDS
        (pl.col("passing_tds").sum() +
         pl.col("rushing_tds").sum()).alias("total_tds"),

        pl.col("passing_yards").sum().alias("total_passing_yards"),
        pl.col("completions").sum().alias("total_completions"),
        pl.col("attempts").sum().alias("total_pass_attempts"),
        pl.col("carries").sum().alias("total_rush_attempts"),
        pl.col("rushing_yards").sum().alias("total_rushing_yards"),
        pl.col("passing_tds").sum().alias("total_passing_tds"),
        pl.col("rushing_tds").sum().alias("total_rushing_tds"),
        pl.col("passing_epa").sum().alias("total_passing_epa"),
        pl.col("rushing_epa").sum().alias("total_rushing_epa"),
        pl.col("passing_interceptions").sum().alias("total_interceptions"),
        pl.col("sacks_suffered").sum().alias("total_sacks_allowed"),
        pl.col("fumbles_lost_total").sum().alias("total_fumbles_lost"),
        pl.col("sack_yards_lost").sum().alias("total_sack_yards_lost"),
    ])
    .sort(["season", "team"])
)



# Added Yards Per Game
reg_off_season = reg_off_season.with_columns(
    (pl.col("total_yards") / pl.col("games_played"))
    .alias("yards_per_game")
)

# Adding Passing ypg, Rushing ypg, total td per game
reg_off_season = reg_off_season.with_columns([
    (
        pl.col("total_passing_yards") / pl.col("games_played")
    ).alias("passing_yards_per_game"),

    (
        pl.col("total_rushing_yards") / pl.col("games_played")
    ).alias("rushing_yards_per_game"),

    (
        pl.col("total_tds") / pl.col("games_played")
    ).alias("tds_per_game")
])

# Adding Completion pct
reg_off_season = reg_off_season.with_columns(
    (
        pl.col("total_completions") /
        pl.col("total_pass_attempts") * 100
    ).alias("completion_pct")
)

# EPA PER GAME
reg_off_season = reg_off_season.with_columns([
    (
        pl.col("total_passing_epa") /
        pl.col("games_played")
    ).alias("passing_epa_per_game"),

    (
        pl.col("total_rushing_epa") /
        pl.col("games_played")
    ).alias("rushing_epa_per_game")
])

# Calculate Interception per game + sacks per game
reg_off_season = reg_off_season.with_columns([
    (
        pl.col("total_interceptions") /
        pl.col("games_played")
    ).alias("interceptions_per_game"),

    (
        pl.col("total_sacks_allowed") /
        pl.col("games_played")
    ).alias("sacks_allowed_per_game")
])


# Total Turnovers | Fumbles Per Game | Turnovers Per Game
reg_off_season = reg_off_season.with_columns([
    (
        pl.col("total_interceptions") +
        pl.col("total_fumbles_lost")
    ).alias("total_turnovers"),

    (
        pl.col("total_fumbles_lost") /
        pl.col("games_played")
    ).alias("fumbles_lost_per_game"),

    (
        (pl.col("total_interceptions") +
         pl.col("total_fumbles_lost")) /
        pl.col("games_played")
    ).alias("turnovers_per_game")
])

# Calculating Yards per play
reg_off_season = reg_off_season.with_columns(
    (
        pl.col("total_yards") /
        (
            pl.col("total_pass_attempts") +
            pl.col("total_rush_attempts") +
            pl.col("total_sacks_allowed")
        )
    ).alias("yards_per_play")
)

# Getting offensive points scored
offensive_points = reg_team.select([
    "season",
    "team",
    "game_id"
]).join(
    team_scores.select([
        "game_id",
        "team",
        "points_scored"
    ]),
    on=["game_id", "team"],
    how="left"
)

# Check for missing scores
print(
    "Missing offensive scores:",
    offensive_points.filter(
        pl.col("points_scored").is_null()
    ).height
)

# Total points scored per team per season
points_scored_season = (
    offensive_points
    .group_by(["season", "team"])
    .agg([
        pl.col("points_scored").sum().alias("total_points_scored")
    ])
)

# Add total points scored to offensive season data
reg_off_season = reg_off_season.join(
    points_scored_season,
    on=["season", "team"],
    how="left"
)

# Calculate points scored per game
reg_off_season = reg_off_season.with_columns(
    (
        pl.col("total_points_scored") /
        pl.col("games_played")
    ).alias("points_scored_per_game")
)

# Verify results
print(reg_off_season.shape)

print(reg_off_season.select([
    "season",
    "team",
    "games_played",
    "total_points_scored",
    "points_scored_per_game"
]).head(10))
# Cleaned offensive df
# print(reg_off_season.select([
#     "Fillinhere"
# ]).head())

# Null Count in Offensive Cleaned Data
# print(reg_off_season.shape)
# print(reg_off_season.null_count())

# Defenses Regular Season
reg_def_weekly = reg_team.select([
    "season",
    "week",
    "team",
    "game_id",
    "opponent_team",
    "def_sacks",
    "def_interceptions",
    "def_fumbles_forced",
    "def_qb_hits",
    "def_tackles_for_loss"
])

opponent_offense = reg_team.select([
    "game_id",

    pl.col("team").alias("opponent_team"),

    # Total net yards allowed
    (
        pl.col("passing_yards") +
        pl.col("rushing_yards") +
        pl.col("sack_yards_lost")
    ).alias("yards_allowed"),

    # Opponent EPA
        pl.col("passing_epa").alias("passing_epa_allowed"),
        pl.col("rushing_epa").alias("rushing_epa_allowed"),

    # Individual passing statistics allowed
    pl.col("passing_yards").alias("passing_yards_allowed"),
    pl.col("passing_tds").alias("passing_tds_allowed"),

    # Individual rushing statistics allowed
    pl.col("rushing_yards").alias("rushing_yards_allowed"),
    pl.col("rushing_tds").alias("rushing_tds_allowed"),

# Total offensive plays by the opponent
(
    pl.col("attempts") +
    pl.col("carries") +
    pl.col("sacks_suffered")
).alias("opponent_offensive_plays"),
])

# Logic is using the opponents total yards as the yards_per_game
reg_def_weekly = reg_def_weekly.join(
    opponent_offense,
    on=["game_id", "opponent_team"],
    how="left"
)

reg_def_weekly = reg_def_weekly.join(
    team_scores.select([
        "game_id",
        "team",
        "points_allowed"
    ]),
    on=["game_id", "team"],
    how="left"
)

reg_def_season = (
    reg_def_weekly
    .group_by(["season", "team"])
    .agg([
        pl.len().alias("games_played"),

        pl.col("points_allowed").sum().alias("total_points_allowed"),

        # Passing stats allowed
        pl.col("passing_yards_allowed").sum().alias("total_passing_yards_allowed"),
        pl.col("passing_tds_allowed").sum().alias("total_passing_tds_allowed"),

        # Rushing stats allowed
        pl.col("rushing_yards_allowed").sum().alias("total_rushing_yards_allowed"),
        pl.col("rushing_tds_allowed").sum().alias("total_rushing_tds_allowed"),

        pl.col("yards_allowed").sum().alias("total_yards_allowed"),

        pl.col("passing_epa_allowed").sum().alias("total_passing_epa_allowed"),

        pl.col("rushing_epa_allowed").sum().alias("total_rushing_epa_allowed"),

        pl.col("def_sacks").sum().alias("total_def_sacks"),

        pl.col("opponent_offensive_plays").sum().alias("total_opponent_plays"),

        pl.col("def_interceptions").sum().alias("total_def_interceptions"),

        pl.col("def_fumbles_forced").sum().alias("total_forced_fumbles"),

        pl.col("def_qb_hits").sum().alias("total_qb_hits"),

        pl.col("def_tackles_for_loss").sum().alias("total_tackles_for_loss")
    ])
    .sort(["season", "team"])
)

# Defensive Points allowed
reg_def_season = reg_def_season.with_columns(
    (
        pl.col("total_points_allowed") /
        pl.col("games_played")
    ).alias("points_allowed_per_game")
)
reg_def_season = reg_def_season.with_columns(
    (
        pl.col("total_yards_allowed") /
        pl.col("games_played")
    ).alias("yards_allowed_per_game")
)


# Checking for Missing Values
missing_def = reg_def_weekly.filter(
    pl.col("yards_allowed").is_null()
)

reg_def_season = reg_def_season.with_columns([

    # Passing yards allowed per game
    (
        pl.col("total_passing_yards_allowed") /
        pl.col("games_played")
    ).alias("passing_yards_allowed_per_game"),

    # Rushing yards allowed per game
    (
        pl.col("total_rushing_yards_allowed") /
        pl.col("games_played")
    ).alias("rushing_yards_allowed_per_game"),

    # Passing TDs allowed per game
    (
        pl.col("total_passing_tds_allowed") /
        pl.col("games_played")
    ).alias("passing_tds_allowed_per_game"),

    # Rushing TDs allowed per game
    (
        pl.col("total_rushing_tds_allowed") /
        pl.col("games_played")
    ).alias("rushing_tds_allowed_per_game")

])

# Adding defensive yards per play
reg_def_season = reg_def_season.with_columns(
    (
        pl.col("total_yards_allowed") /
        pl.col("total_opponent_plays")
    ).alias("yards_allowed_per_play")
)

# Calculating Defensive Passing + Rushing EPA (Lower Epa means a better denfensive performance)
reg_def_season = reg_def_season.with_columns([

    (
        pl.col("total_passing_epa_allowed") /
        pl.col("games_played")
    ).alias("passing_epa_allowed_per_game"),

    (
        pl.col("total_rushing_epa_allowed") /
        pl.col("games_played")
    ).alias("rushing_epa_allowed_per_game")

])

# Total Offensive EPA allowed
reg_def_season = reg_def_season.with_columns(
    (
        pl.col("total_passing_epa_allowed") +
        pl.col("total_rushing_epa_allowed")
    ).alias("total_offensive_epa_allowed")
)


# Defensive EPA per play
reg_def_season = reg_def_season.with_columns(
    (
        pl.col("total_offensive_epa_allowed") /
        pl.col("total_opponent_plays")
    ).alias("estimated_epa_allowed_per_play")
)

print(reg_def_season.shape)

print(reg_def_season.select([
    "season",
    "team",
    "games_played",
    "total_points_allowed",
    "points_allowed_per_game",
    "yards_allowed_per_game",
    "yards_allowed_per_play"
]).head(10))

# ----------------------------------------------------------------------
# Regular Season (2015-2025) Offense stats - > reg_off_season
# Regular Season (2015-2025) Defense stats - > reg_def_season

# Join Datasets
# Combining offensive and defensive season data
full_season = reg_off_season.join(
    reg_def_season.drop("games_played"),
    on=["season", "team"],
    how="inner",
    validate="1:1"
).sort(["season", "team"])



# Creating the Outcomes ---------------------------------------------------------
# Get only postseason games
playoff_games = schedules.filter(
    pl.col("game_type").is_in([
        "WC",   # Wild Card
        "DIV",  # Divisional Round
        "CON",  # Conference Championship
        "SB"    # Super Bowl
    ])
)

# Create a record for each team in every playoff game
playoff_team_games = pl.concat([

    # Home teams
    playoff_games.select([
        "season",
        "game_type",
        pl.col("home_team").replace(team_names).alias("team"),

        (
            pl.col("home_score") >
            pl.col("away_score")
        ).alias("won_game")
    ]),

    # Away teams
    playoff_games.select([
        "season",
        "game_type",
        pl.col("away_team").replace(team_names).alias("team"),

        (
            pl.col("away_score") >
            pl.col("home_score")
        ).alias("won_game")
    ])

])

# Assign a postseason classification to each playoff game
playoff_team_games = playoff_team_games.with_columns(

    pl.when(
        (pl.col("game_type") == "SB") &
        (pl.col("won_game"))
    ).then(5)

    .when(
        (pl.col("game_type") == "SB") &
        (~pl.col("won_game"))
    ).then(4)

    .when(
        (pl.col("game_type") == "CON") &
        (~pl.col("won_game"))
    ).then(3)

    .otherwise(2)

    .alias("playoff_class")
)

# Find the highest postseason result for each team
playoff_outcomes = (
    playoff_team_games
    .group_by(["season", "team"])
    .agg([
        pl.col("playoff_class").max().alias("playoff_class")
    ])
    .sort(["season", "team"])
)

# Add postseason outcome to the full dataset
full_season = full_season.join(
    playoff_outcomes,
    on=["season", "team"],
    how="left",
    validate="1:1"
)

# Teams without a playoff record missed the playoffs
full_season = full_season.with_columns(
    pl.col("playoff_class")
    .fill_null(1)
    .cast(pl.Int8)
)

print(full_season.shape)

print(
    full_season
    .group_by("playoff_class")
    .len()
    .sort("playoff_class")
)
# ---------------------------------------------------------------------
# Verify postseason classification for 2015
# Won Super Bowl = 5
# Reached but Lost Super Bowl = 4
# Reached the AFC/NFC Championship = 3
# Made Playsoffs but lost in wildcard/divisional = 2
# Missed Playoffs = 1

# Checking to See if Teams are classified correctly
print(
    full_season
    .filter(
        (pl.col("season") == 2015) &
        (pl.col("team").is_in([
            "DEN", "CAR", "ARI", "NE", "CLE"
        ]))
    )
    .select([
        "season",
        "team",
        "playoff_class"
    ])
    .sort("playoff_class", descending=True)
)

# Check playoff classification distribution
print(
    full_season
    .group_by("playoff_class")
    .len()
    .sort("playoff_class")

)# FINAL DATA QUALITY CHECKS

# 1. Duplicate team-seasons
# duplicates = (
#     full_season
#     .group_by(["season", "team"])
#     .len()
#     .filter(pl.col("len") > 1)
# )

print("Duplicate team-seasons:", duplicates.height)


# 2. Total missing values
# print(
#     "Missing values:",
#     sum(full_season.null_count().row(0))
# )


# 3. Check numerical columns for NaN and infinity
# float_cols = [
#     name for name, dtype in full_season.schema.items()
#     if dtype in (pl.Float32, pl.Float64)
# ]
#
# invalid_rows = full_season.filter(
#     pl.any_horizontal([
#         ~pl.col(col).is_finite()
#         for col in float_cols
#     ])
# )
#
# print("Rows with NaN or infinity:", invalid_rows.height)

# Results
# Duplicate team-seasons: 0
# Missing values: 0
# Rows with NaN or infinity: 0



