import os
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from src.config import MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD, SEASON
from src.model_utils import _load_sched, build_training_table

DEFAULT_SEASONS = [SEASON - 4, SEASON - 3, SEASON - 2, SEASON - 1]
MIN_TRAIN_ROWS = int(os.getenv("BACKTEST_MIN_TRAIN_ROWS", "30"))


def parse_seasons() -> List[int]:
    raw = os.getenv("BACKTEST_SEASONS", "").strip()

    if raw:
        return sorted({int(x) for x in raw.split(",") if x.strip()})

    return sorted({int(x) for x in DEFAULT_SEASONS if x >= 1990})


def american_to_implied_prob_array(odds):
    odds = np.asarray(odds, dtype=float)
    out = np.full(odds.shape, 0.5, dtype=float)

    pos = odds > 0
    neg = odds < 0

    with np.errstate(divide="ignore", invalid="ignore"):
        out[pos] = 100.0 / (odds[pos] + 100.0)
        out[neg] = (-odds[neg]) / ((-odds[neg]) + 100.0)

    out[~np.isfinite(out)] = 0.5
    return out


def remove_vig_array(prob_home, prob_away):
    prob_home = np.asarray(prob_home, dtype=float)
    prob_away = np.asarray(prob_away, dtype=float)

    total = prob_home + prob_away

    with np.errstate(divide="ignore", invalid="ignore"):
        fair_home = np.where(total > 0, prob_home / total, 0.5)
        fair_away = np.where(total > 0, prob_away / total, 0.5)

    fair_home[~np.isfinite(fair_home)] = 0.5
    fair_away[~np.isfinite(fair_away)] = 0.5

    return fair_home, fair_away


def profit_units_array(odds, stake=1.0):
    odds = np.asarray(odds, dtype=float)
    out = np.zeros(odds.shape, dtype=float)

    pos = odds > 0
    neg = odds < 0

    out[pos] = stake * odds[pos] / 100.0
    out[neg] = stake * 100.0 / (-odds[neg])

    return out


def backtest_fold(test_season, all_df, feature_cols):
    train = (
        all_df[all_df["season"] < test_season]
        .dropna(subset=feature_cols + ["home_margin"])
        .copy()
    )

    test = (
        all_df[
            (all_df["season"] == test_season)
            & all_df["spread_line"].notna()
        ]
        .dropna(
            subset=feature_cols
            + [
                "home_margin",
                "home_spread_odds",
                "away_spread_odds",
            ]
        )
        .copy()
    )

    if len(train) < MIN_TRAIN_ROWS or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    X = train[feature_cols].to_numpy(dtype=float)
    y = train["home_margin"].to_numpy(dtype=float)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    resid = y - model.predict(X)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    Xt = test[feature_cols].to_numpy(dtype=float)
    mu = model.predict(Xt)

    spread = test["spread_line"].to_numpy(dtype=float)
    home_odds = test["home_spread_odds"].to_numpy(dtype=float)
    away_odds = test["away_spread_odds"].to_numpy(dtype=float)
    actual_margin = test["home_margin"].to_numpy(dtype=float)

    home_teams = test["home_team"].to_numpy()
    away_teams = test["away_team"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        p_home = norm.cdf((mu + spread) / sigma)

    p_home = np.clip(p_home, 0.0, 1.0)
    p_away = 1.0 - p_home

    imp_home = american_to_implied_prob_array(home_odds)
    imp_away = american_to_implied_prob_array(away_odds)

    fair_home, fair_away = remove_vig_array(imp_home, imp_away)

    edge_home = p_home - fair_home
    edge_away = p_away - fair_away

    pick_home = edge_home >= edge_away

    edge = np.where(pick_home, edge_home, edge_away)
    model_p = np.where(pick_home, p_home, p_away)
    fair_p = np.where(pick_home, fair_home, fair_away)

    pick_line = np.where(pick_home, spread, -spread)
    pick_odds = np.where(pick_home, home_odds, away_odds)
    pick_side = np.where(pick_home, "HOME", "AWAY")
    pick_team = np.where(pick_home, home_teams, away_teams)

    home_cover = (actual_margin + spread) > 0
    away_cover = (actual_margin + spread) < 0
    push = np.isclose(actual_margin + spread, 0.0, atol=1e-9)

    win = np.where(pick_home, home_cover, away_cover)

    result = np.where(
        push,
        "PUSH",
        np.where(win, "WIN", "LOSS"),
    )

    roi = np.where(
        result == "WIN",
        profit_units_array(pick_odds, 1.0),
        np.where(result == "LOSS", -1.0, 0.0),
    )

    out = pd.DataFrame(
        {
            "game_id": test["game_id"].to_numpy(),
            "season": test["season"].to_numpy(),
            "week": test["week"].to_numpy(),
            "home_team": home_teams,
            "away_team": away_teams,
            "spread_line": spread,
            "home_spread_odds": home_odds,
            "away_spread_odds": away_odds,
            "actual_home_margin": actual_margin,
            "mu_home_margin": mu,
            "sigma": sigma,
            "pick_team": pick_team,
            "pick_side": pick_side,
            "pick_line": pick_line,
            "pick_odds": pick_odds,
            "model_probability": model_p,
            "implied_probability_no_vig": fair_p,
            "edge": edge,
            "pick_result": result,
            "roi_units": roi,
        }
    )

    fav_valid = spread != 0

    if not np.any(fav_valid):
        fav = out.iloc[0:0].copy()
        fav["baseline"] = "favorite"
        return out, fav

    fav_home = spread < 0

    fav_odds = np.where(fav_home, home_odds, away_odds)
    fav_win = np.where(fav_home, home_cover, away_cover)

    fav_result = np.where(
        push,
        "PUSH",
        np.where(fav_win, "WIN", "LOSS"),
    )

    fav_roi = np.where(
        fav_result == "WIN",
        profit_units_array(fav_odds, 1.0),
        np.where(fav_result == "LOSS", -1.0, 0.0),
    )

    fav = out[fav_valid].copy()
    fav["baseline"] = "favorite"
    fav["pick_team"] = np.where(fav_home, home_teams, away_teams)[fav_valid]
    fav["pick_side"] = np.where(fav_home, "HOME", "AWAY")[fav_valid]
    fav["pick_odds"] = fav_odds[fav_valid]
    fav["pick_result"] = fav_result[fav_valid]
    fav["roi_units"] = fav_roi[fav_valid]

    return out, fav


def summarize(label, df):
    if df is None or df.empty:
        return pd.DataFrame(
            [
                {
                    "group": label,
                    "bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "hit_rate": np.nan,
                    "roi_units": 0.0,
                    "roi_per_non_push_bet": np.nan,
                    "avg_edge": np.nan,
                    "avg_model_prob": np.nan,
                    "avg_fair_prob": np.nan,
                }
            ]
        )

    bets = len(df)
    wins = int((df["pick_result"] == "WIN").sum())
    losses = int((df["pick_result"] == "LOSS").sum())
    pushes = int((df["pick_result"] == "PUSH").sum())
    non_push = bets - pushes

    hit_rate = wins / non_push if non_push > 0 else np.nan
    roi_units = float(df["roi_units"].sum())
    roi_per_non_push = roi_units / non_push if non_push > 0 else np.nan

    avg_edge = (
        float(df["edge"].mean())
        if "edge" in df.columns and df["edge"].notna().any()
        else np.nan
    )

    avg_model = (
        float(df["model_probability"].mean())
        if "model_probability" in df.columns
        and df["model_probability"].notna().any()
        else np.nan
    )

    avg_fair = (
        float(df["implied_probability_no_vig"].mean())
        if "implied_probability_no_vig" in df.columns
        and df["implied_probability_no_vig"].notna().any()
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "group": label,
                "bets": bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": hit_rate,
                "roi_units": roi_units,
                "roi_per_non_push_bet": roi_per_non_push,
                "avg_edge": avg_edge,
                "avg_model_prob": avg_model,
                "avg_fair_prob": avg_fair,
            }
        ]
    )


def calibration_table(df):
    if df is None or df.empty or "edge" not in df.columns:
        return pd.DataFrame()

    out = df.copy()

    bins = [-1.0, 0.02, 0.04, 0.06, 0.08, 0.10, 2.0]
    labels = ["<2%", "2-4%", "4-6%", "6-8%", "8-10%", ">10%"]

    out["edge_bucket"] = pd.cut(
        out["edge"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    rows = []

    for bucket, g in out.groupby("edge_bucket", observed=True):
        bets = len(g)
        wins = int((g["pick_result"] == "WIN").sum())
        losses = int((g["pick_result"] == "LOSS").sum())
        pushes = int((g["pick_result"] == "PUSH").sum())
        non_push = bets - pushes

        actual_win_rate = wins / non_push if non_push > 0 else np.nan
        avg_model = float(g["model_probability"].mean()) if "model_probability" in g.columns else np.nan
        avg_fair = float(g["implied_probability_no_vig"].mean()) if "implied_probability_no_vig" in g.columns else np.nan
        avg_edge = float(g["edge"].mean()) if "edge" in g.columns else np.nan
        roi_units = float(g["roi_units"].sum())

        rows.append(
            {
                "edge_bucket": str(bucket),
                "bets": bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "actual_win_rate": actual_win_rate,
                "avg_model_probability": avg_model,
                "avg_fair_probability": avg_fair,
                "avg_edge": avg_edge,
                "calibration_gap": actual_win_rate - avg_model if non_push > 0 else np.nan,
                "roi_units": roi_units,
            }
        )

    return pd.DataFrame(rows)


def main():
    seasons = parse_seasons()

    print("Backtest seasons: " + str(seasons))

    if len(seasons) < 2:
        raise SystemExit("Need at least two seasons for walk-forward backtest.")

    sched = _load_sched(seasons)

    if sched.empty:
        raise SystemExit("nflverse returned no schedule data for these seasons.")

    all_df, feature_cols = build_training_table(seasons, sched=sched)

    if all_df.empty:
        raise SystemExit("Training table is empty.")

    needed = [
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]

    missing = [c for c in needed if c not in all_df.columns]

    if missing:
        print("Missing historical betting columns: " + str(missing))
        print("Available columns: " + str(list(all_df.columns)))
        raise SystemExit("Cannot run historical backtest without spread and odds columns.")

    numeric_cols = (
        needed
        + feature_cols
        + [
            "home_margin",
            "season",
            "week",
        ]
    )

    for col in numeric_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    all_df = all_df.dropna(
        subset=feature_cols
        + [
            "home_margin",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
        ]
    )

    if all_df.empty:
        raise SystemExit("No usable backtest rows after cleaning.")

    test_seasons = sorted(all_df["season"].dropna().unique())[1:]

    print("Features: " + str(feature_cols))
    print("Usable rows: " + str(len(all_df)))
    print("Walk-forward test seasons: " + str([int(x) for x in test_seasons]))

    picks_list = []
    fav_list = []
    season_rows = []

    for ts in test_seasons:
        ts_int = int(ts)

        picks, fav = backtest_fold(ts_int, all_df, feature_cols)

        if picks.empty:
            print("Skipping test season " + str(ts_int) + ": insufficient train/test data.")
            continue

        picks["fold_test_season"] = ts_int
        fav["fold_test_season"] = ts_int

        picks_list.append(picks)
        fav_list.append(fav)

        model_threshold = picks[picks["edge"] >= MIN_EDGE_THRESHOLD]

        season_rows.append(summarize(str(ts_int) + " model_threshold", model_threshold))
        season_rows.append(summarize(str(ts_int) + " favorite", fav))

    if not picks_list:
        raise SystemExit("No backtest folds produced results.")

    picks = pd.concat(picks_list, ignore_index=True)
    fav = pd.concat(fav_list, ignore_index=True)

    model_threshold = picks[picks["edge"] >= MIN_EDGE_THRESHOLD].copy()
    model_auto = model_threshold[model_threshold["edge"] <= MAX_AUTO_EDGE].copy()
    large_edge = model_threshold[model_threshold["edge"] > MAX_AUTO_EDGE].copy()

    os.makedirs("data", exist_ok=True)

    picks.to_csv("data/backtest_all_predictions.csv", index=False)
    model_threshold.to_csv("data/backtest_model_threshold_picks.csv", index=False)

    print("\n=== Season summaries ===")
    print(pd.concat(season_rows, ignore_index=True).to_string(index=False))

    print("\n=== Overall summaries ===")
    overall = pd.concat(
        [
            summarize("all_forced_model_picks", picks),
            summarize("model_threshold_picks", model_threshold),
            summarize("model_auto_band", model_auto),
            summarize("favorite_baseline", fav),
        ],
        ignore_index=True,
    )
    print(overall.to_string(index=False))

    print("\n=== Calibration for model threshold picks ===")
    cal = calibration_table(model_threshold)

    if cal.empty:
        print("No calibration rows.")
    else:
        print(cal.to_string(index=False))

    if not large_edge.empty:
        print("\n=== Large-edge picks above MAX_AUTO_EDGE=" + str(MAX_AUTO_EDGE) + " ===")
        print(
            large_edge[
                [
                    "fold_test_season",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "pick_team",
                    "pick_side",
                    "pick_line",
                    "pick_odds",
                    "model_probability",
                    "implied_probability_no_vig",
                    "edge",
                    "pick_result",
                    "roi_units",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print("  data/backtest_all_predictions.csv")
    print("  data/backtest_model_threshold_picks.csv")

    print("\nImportant:")
    print("This backtest uses nflverse historical closing lines, not opening lines.")
    print("It evaluates model edge vs market close.")
    print("Live Google Sheet CLV remains the prospective opening-to-close metric.")


if __name__ == "__main__":
    main()
