import numpy as np
import pandas as pd

from src.data_loader import load_datasets, build_lookups
from src.prediction import simulate_match


def prepare_group_fixtures():
    """Add group labels to group-stage fixtures."""

    data = load_datasets()
    lookups = build_lookups()

    df_fixtures = data["fixtures"].copy()
    team_to_group = lookups["team_to_group"]

    df_fixtures["group"] = df_fixtures["home_team"].map(team_to_group)

    return df_fixtures


def create_empty_group_table(group_teams):
    """Create an empty table for one group."""

    return pd.DataFrame({
        "team": group_teams,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    })


def update_group_table(table, match):
    """Update a group table after one match."""

    table = table.copy()

    home_team = match["home_team"]
    away_team = match["away_team"]
    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    table.loc[table["team"] == home_team, "played"] += 1
    table.loc[table["team"] == away_team, "played"] += 1

    table.loc[table["team"] == home_team, "goals_for"] += home_goals
    table.loc[table["team"] == home_team, "goals_against"] += away_goals

    table.loc[table["team"] == away_team, "goals_for"] += away_goals
    table.loc[table["team"] == away_team, "goals_against"] += home_goals

    if home_goals > away_goals:
        table.loc[table["team"] == home_team, "wins"] += 1
        table.loc[table["team"] == away_team, "losses"] += 1
        table.loc[table["team"] == home_team, "points"] += 3

    elif away_goals > home_goals:
        table.loc[table["team"] == away_team, "wins"] += 1
        table.loc[table["team"] == home_team, "losses"] += 1
        table.loc[table["team"] == away_team, "points"] += 3

    else:
        table.loc[table["team"] == home_team, "draws"] += 1
        table.loc[table["team"] == away_team, "draws"] += 1
        table.loc[table["team"] == home_team, "points"] += 1
        table.loc[table["team"] == away_team, "points"] += 1

    table["goal_difference"] = table["goals_for"] - table["goals_against"]

    return table


def rank_group_table(table):
    """Rank teams inside one group."""

    table = table.copy()
    table["random_tiebreaker"] = np.random.random(len(table))

    table = (
        table
        .sort_values(
            by=["points", "goal_difference", "goals_for", "random_tiebreaker"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )

    table["group_rank"] = table.index + 1

    return table.drop(columns=["random_tiebreaker"])


def simulate_group(group_name):
    """Simulate all matches in one group."""

    data = load_datasets()
    df_groups = data["groups"]
    df_group_fixtures = prepare_group_fixtures()

    group_teams = (
        df_groups[df_groups["group"] == group_name]
        .sort_values("position")["nation"]
        .tolist()
    )

    group_matches = df_group_fixtures[df_group_fixtures["group"] == group_name]

    table = create_empty_group_table(group_teams)
    simulated_matches = []

    for _, row in group_matches.iterrows():
        match = simulate_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            neutral=bool(row["neutral"]),
        )

        simulated_matches.append(match)
        table = update_group_table(table, match)

    ranked_table = rank_group_table(table)
    ranked_table["group"] = group_name

    return ranked_table, pd.DataFrame(simulated_matches)


def simulate_group_stage():
    """Simulate all 12 groups."""

    data = load_datasets()
    df_groups = data["groups"]

    all_group_tables = []
    all_group_matches = []

    for group_name in sorted(df_groups["group"].unique()):
        group_table, group_matches = simulate_group(group_name)

        all_group_tables.append(group_table)
        all_group_matches.append(group_matches)

    df_group_tables = pd.concat(all_group_tables, ignore_index=True)
    df_group_matches = pd.concat(all_group_matches, ignore_index=True)

    return df_group_tables, df_group_matches
import numpy as np
import pandas as pd

from src.prediction import predict_match


# Knockout rounds are played in this order. Each round resolves its
# home/away teams from the winners (W...) or runners-up (RU...) of
# earlier matches.
KNOCKOUT_ROUND_ORDER = ["R16", "QF", "SF", "3rd", "Final"]


def simulate_knockout_match(home_team, away_team, round_name, neutral=True):
    """
    Simulate one knockout match. Draws are not allowed, so a tie after
    regulation is decided by a penalty shootout, weighted by each team's
    relative win probability from the Poisson model.
    """

    pred = predict_match(home_team, away_team, neutral=neutral)

    home_goals = int(np.random.poisson(pred["home_xg"]))
    away_goals = int(np.random.poisson(pred["away_xg"]))

    if home_goals > away_goals:
        winner, loser = home_team, away_team
        decided_by = "regulation"
    elif away_goals > home_goals:
        winner, loser = away_team, home_team
        decided_by = "regulation"
    else:
        # Penalty shootout: split the draw mass between the two teams
        # according to their relative win probabilities.
        p_home = pred["home_win_prob"]
        p_away = pred["away_win_prob"]
        total = p_home + p_away

        prob_home_wins = 0.5 if total == 0 else p_home / total

        if np.random.random() < prob_home_wins:
            winner, loser = home_team, away_team
        else:
            winner, loser = away_team, home_team

        decided_by = "penalties"

    return {
        "round": round_name,
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": pred["home_xg"],
        "away_xg": pred["away_xg"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": winner,
        "loser": loser,
        "decided_by": decided_by,
    }


def _resolve_slot(slot, match_winner, match_loser):
    """
    Resolve a knockout slot reference to a real team.

    - "W74"   -> winner of match M74
    - "RU101" -> runner-up (loser) of match M101
    """

    slot = str(slot)

    if slot.startswith("RU"):
        return match_loser[f"M{slot[2:]}"]

    if slot.startswith("W"):
        return match_winner[f"M{slot[1:]}"]

    raise ValueError(f"Cannot resolve knockout slot: {slot}")


def simulate_knockout_stage(df_knockout, df_round_32_filled):
    """
    Simulate the full knockout stage (R32 -> Final).

    Returns:
        df_knockout_results: one row per knockout match
        winner: tournament winner
        runner_up: losing finalist
    """

    df_knockout = df_knockout.copy()

    match_winner = {}
    match_loser = {}
    results = []

    # Round of 32 already has real teams assigned by the bracket builder.
    for _, row in df_round_32_filled.iterrows():
        match_id = row["match_id"]

        res = simulate_knockout_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            round_name="R32",
        )
        res["match_id"] = match_id

        match_winner[match_id] = res["winner"]
        match_loser[match_id] = res["loser"]
        results.append(res)

    # Remaining rounds resolve their teams from earlier match results.
    for round_name in KNOCKOUT_ROUND_ORDER:
        round_rows = df_knockout[df_knockout["round"] == round_name]

        for _, row in round_rows.iterrows():
            match_id = row["match_id"]

            home_team = _resolve_slot(row["home_slot"], match_winner, match_loser)
            away_team = _resolve_slot(row["away_slot"], match_winner, match_loser)

            res = simulate_knockout_match(
                home_team=home_team,
                away_team=away_team,
                round_name=round_name,
            )
            res["match_id"] = match_id

            match_winner[match_id] = res["winner"]
            match_loser[match_id] = res["loser"]
            results.append(res)

    df_knockout_results = pd.DataFrame(results)

    final_match = df_knockout_results[df_knockout_results["round"] == "Final"].iloc[0]
    winner = final_match["winner"]
    runner_up = final_match["loser"]

    return df_knockout_results, winner, runner_up