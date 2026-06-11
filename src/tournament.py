from collections import Counter

from src.data_loader import load_datasets
from src.group_stage import simulate_group_stage
from src.qualification import get_qualified_teams
from src.bracket import get_round_32_template, build_round_32_bracket
from src.knockout import simulate_knockout_stage


def simulate_tournament():
    """Simulate one full World Cup tournament."""

    data = load_datasets()
    df_knockout = data["knockout"]

    df_group_tables, df_group_matches = simulate_group_stage()

    direct_qualifiers, best_third_placed, qualified_teams = get_qualified_teams(
        df_group_tables
    )

    df_round_32 = get_round_32_template()

    df_round_32_filled = build_round_32_bracket(
        df_round_32=df_round_32,
        direct_qualifiers=direct_qualifiers,
        best_third_placed=best_third_placed,
    )

    df_knockout_results, winner, runner_up = simulate_knockout_stage(
        df_knockout=df_knockout,
        df_round_32_filled=df_round_32_filled,
    )

    r32_teams = qualified_teams

    r16_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "R32", "winner"]
        .tolist()
    )

    qf_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "R16", "winner"]
        .tolist()
    )

    sf_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "QF", "winner"]
        .tolist()
    )

    final_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "SF", "winner"]
        .tolist()
    )

    summary = {
        "winner": winner,
        "runner_up": runner_up,
        "r32_teams": r32_teams,
        "r16_teams": r16_teams,
        "qf_teams": qf_teams,
        "sf_teams": sf_teams,
        "final_teams": final_teams,
    }

    return {
        "summary": summary,
        "group_tables": df_group_tables,
        "group_matches": df_group_matches,
        "round_32_bracket": df_round_32_filled,
        "knockout_results": df_knockout_results,
    }


def run_aggregated_simulations(n_simulations=200, progress_callback=None):
    """
    Run many full-tournament simulations and aggregate the results.

    Because the group-stage fixtures are identical in every run, their
    scores can be averaged directly. Knockout matchups change each run,
    so those are summarised as per-team round-reach probabilities.

    Returns a dict with:
        n_simulations
        group_match_summary : one consensus row per group fixture
        winner_counts       : Counter of champions across all runs
        round_reach         : per-team probability of reaching each round
    """

    import pandas as pd

    # fixture key -> list of (home_goals, away_goals) across all runs
    fixture_scores = {}
    fixture_order = []

    winner_counts = Counter()

    # round name -> Counter of how many runs each team reached it
    reach_counters = {
        "r16": Counter(),
        "qf": Counter(),
        "sf": Counter(),
        "final": Counter(),
        "winner": Counter(),
    }

    # team -> accumulated group-stage stats across all runs
    group_stats = {}
    team_to_group = {}

    # round -> list of all matches across all runs
    knockout_matches = {
        "R32": [],
        "R16": [],
        "QF": [],
        "SF": [],
        "Final": [],
    }

    # Store a representative tournament (the one with the most frequent champion)
    representative_tournament = None

    for i in range(n_simulations):
        tournament = simulate_tournament()
        summary = tournament["summary"]
        group_matches = tournament["group_matches"]
        group_tables = tournament["group_tables"]
        knockout_results = tournament["knockout_results"]

        # Keep the first tournament with the eventual most frequent champion
        if representative_tournament is None:
            representative_tournament = tournament

        for _, gt in group_tables.iterrows():
            team = gt["team"]
            team_to_group[team] = gt["group"]

            stats = group_stats.setdefault(team, {
                "points": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "qualified_count": 0,
            })

            stats["points"] += gt["points"]
            stats["wins"] += gt["wins"]
            stats["draws"] += gt["draws"]
            stats["losses"] += gt["losses"]
            stats["goals_for"] += gt["goals_for"]
            stats["goals_against"] += gt["goals_against"]
            stats["goal_difference"] += gt["goal_difference"]

            if gt["group_rank"] <= 2:
                stats["qualified_count"] += 1

        for _, match in group_matches.iterrows():
            key = (match["home_team"], match["away_team"])

            if key not in fixture_scores:
                fixture_scores[key] = []
                fixture_order.append(key)

            fixture_scores[key].append(
                (int(match["home_goals"]), int(match["away_goals"]))
            )

        winner_counts[summary["winner"]] += 1

        for team in summary["r16_teams"]:
            reach_counters["r16"][team] += 1
        for team in summary["qf_teams"]:
            reach_counters["qf"][team] += 1
        for team in summary["sf_teams"]:
            reach_counters["sf"][team] += 1
        for team in summary["final_teams"]:
            reach_counters["final"][team] += 1
        reach_counters["winner"][summary["winner"]] += 1

        # Track knockout matches
        for _, match in knockout_results.iterrows():
            round_name = match["round"]
            if round_name in knockout_matches:
                knockout_matches[round_name].append({
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "home_goals": int(match["home_goals"]),
                    "away_goals": int(match["away_goals"]),
                    "winner": match["winner"],
                })

        if progress_callback is not None:
            progress_callback(i + 1, n_simulations)

    # ---- aggregate the group fixtures ----
    group_rows = []

    for key in fixture_order:
        home_team, away_team = key
        scores = fixture_scores[key]
        total = len(scores)

        score_counts = Counter(scores)
        (best_home, best_away), best_freq = score_counts.most_common(1)[0]

        home_wins = sum(1 for h, a in scores if h > a)
        draws = sum(1 for h, a in scores if h == a)
        away_wins = sum(1 for h, a in scores if h < a)

        if home_wins >= away_wins and home_wins >= draws:
            predicted_winner = home_team
        elif away_wins >= home_wins and away_wins >= draws:
            predicted_winner = away_team
        else:
            predicted_winner = "Draw"

        # Calculate odds (decimal odds = 1 / probability)
        home_odds = 1 / (home_wins / total) if home_wins > 0 else 0
        draw_odds = 1 / (draws / total) if draws > 0 else 0
        away_odds = 1 / (away_wins / total) if away_wins > 0 else 0

        group_rows.append({
            "home_team": home_team,
            "away_team": away_team,
            "predicted_home_goals": best_home,
            "predicted_away_goals": best_away,
            "scoreline_freq": best_freq / total,
            "avg_home_goals": sum(h for h, _ in scores) / total,
            "avg_away_goals": sum(a for _, a in scores) / total,
            "home_win_pct": home_wins / total,
            "draw_pct": draws / total,
            "away_win_pct": away_wins / total,
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "predicted_winner": predicted_winner,
        })

    group_match_summary = pd.DataFrame(group_rows)

    # ---- per-team round-reach probabilities ----
    all_teams = set()
    for counter in reach_counters.values():
        all_teams.update(counter.keys())

    reach_rows = []
    for team in all_teams:
        reach_rows.append({
            "team": team,
            "r16_prob": reach_counters["r16"][team] / n_simulations,
            "qf_prob": reach_counters["qf"][team] / n_simulations,
            "sf_prob": reach_counters["sf"][team] / n_simulations,
            "final_prob": reach_counters["final"][team] / n_simulations,
            "winner_prob": reach_counters["winner"][team] / n_simulations,
        })

    round_reach = (
        pd.DataFrame(reach_rows)
        .sort_values("winner_prob", ascending=False)
        .reset_index(drop=True)
    )

    # ---- consensus group tables (averaged across runs) ----
    table_rows = []
    for team, stats in group_stats.items():
        table_rows.append({
            "group": team_to_group[team],
            "team": team,
            "avg_points": stats["points"] / n_simulations,
            "avg_wins": stats["wins"] / n_simulations,
            "avg_draws": stats["draws"] / n_simulations,
            "avg_losses": stats["losses"] / n_simulations,
            "avg_goals_for": stats["goals_for"] / n_simulations,
            "avg_goals_against": stats["goals_against"] / n_simulations,
            "avg_goal_difference": stats["goal_difference"] / n_simulations,
            "qualify_prob": stats["qualified_count"] / n_simulations,
        })

    group_table_summary = pd.DataFrame(table_rows)

    # Rank within each group by average points, then goal difference, then goals for.
    group_table_summary = (
        group_table_summary
        .sort_values(
            by=["group", "avg_points", "avg_goal_difference", "avg_goals_for"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

    group_table_summary["consensus_rank"] = (
        group_table_summary.groupby("group").cumcount() + 1
    )
    group_table_summary["qualified"] = group_table_summary["consensus_rank"] <= 2

    # ---- Find the most frequent champion and use their tournament as representative ----
    most_frequent_champion = winner_counts.most_common(1)[0][0]

    # Run one more simulation to get a tournament with the most frequent champion
    # (simpler than tracking all tournaments)
    for _ in range(n_simulations):
        tournament = simulate_tournament()
        if tournament["summary"]["winner"] == most_frequent_champion:
            representative_tournament = tournament
            break
    else:
        # Fallback: use the first tournament if we couldn't find one with the most frequent champion
        representative_tournament = simulate_tournament()

    # ---- aggregate knockout matches ----
    knockout_summary = {}
    for round_name, matches in knockout_matches.items():
        if not matches:
            continue

        # Count unique matchups (sorted team pair to avoid double-counting)
        matchup_counter = Counter()
        score_counter = Counter()
        result_counter = Counter()  # Track home win, away win for the most common matchup

        for m in matches:
            # Sort teams to create a consistent key
            teams = tuple(sorted([m["home_team"], m["away_team"]]))
            matchup_counter[teams] += 1
            score_counter[(teams, m["home_goals"], m["away_goals"])] += 1

        # Find most common matchup for this round
        if matchup_counter:
            top_matchup, matchup_count = matchup_counter.most_common(1)[0]
            team1, team2 = top_matchup

            # Find most common scoreline for this matchup
            matchup_scores = [
                (h, a) for (t, h, a), c in score_counter.items() if t == top_matchup
            ]
            if matchup_scores:
                score_counter_filtered = Counter(matchup_scores)
                (best_home, best_away), score_count = score_counter_filtered.most_common(1)[0]
            else:
                best_home, best_away = 0, 0
                score_count = 0

            # Count results for this specific matchup (team1 vs team2)
            # Need to check which team was home/away in each match
            team1_wins = 0
            team2_wins = 0
            for m in matches:
                if set([m["home_team"], m["away_team"]]) == set([team1, team2]):
                    if m["winner"] == team1:
                        team1_wins += 1
                    elif m["winner"] == team2:
                        team2_wins += 1

            total_matchup_games = team1_wins + team2_wins
            team1_win_pct = team1_wins / total_matchup_games if total_matchup_games > 0 else 0
            team2_win_pct = team2_wins / total_matchup_games if total_matchup_games > 0 else 0

            # Calculate odds
            team1_odds = 1 / team1_win_pct if team1_win_pct > 0 else 0
            team2_odds = 1 / team2_win_pct if team2_win_pct > 0 else 0

            knockout_summary[round_name] = {
                "team1": team1,
                "team2": team2,
                "home_goals": best_home,
                "away_goals": best_away,
                "matchup_freq": matchup_count / len(matches),
                "scoreline_freq": score_count / matchup_count if matchup_count > 0 else 0,
                "team1_win_pct": team1_win_pct,
                "team2_win_pct": team2_win_pct,
                "team1_odds": team1_odds,
                "team2_odds": team2_odds,
            }

    return {
        "n_simulations": n_simulations,
        "group_match_summary": group_match_summary,
        "group_table_summary": group_table_summary,
        "winner_counts": winner_counts,
        "round_reach": round_reach,
        "knockout_summary": knockout_summary,
        "representative_tournament": representative_tournament,
    }


def precompute_and_save(n_simulations=200, output_path=None, progress_callback=None):
    """Run the aggregated simulations once and pickle the results to disk."""

    import pickle
    from pathlib import Path

    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "simulation_results.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_aggregated_simulations(
        n_simulations=n_simulations,
        progress_callback=progress_callback,
    )

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    return output_path