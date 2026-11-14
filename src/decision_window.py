"""When a game may be decided: pure functions, no I/O.

A game is decidable in a run only when it is in the current decision week, kicks
off within the horizon, and every game that should be over has a final score.
See notes/implementation-plan/R1-decision-timing.md.
"""

import pandas as pd

EASTERN = "America/New_York"
MAX_DECISION_DAYS = 7.5
RESULTS_GRACE_HOURS = 5.0


def _utc(value):
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT
    return ts if isinstance(ts, pd.Timestamp) else pd.NaT


def _week(value):
    w = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(w) else int(w)


def current_decision_week(games, now_utc):
    """Smallest week that still has a kickoff after now. games: iterable of (week, kickoff_utc)."""
    now = _utc(now_utc)
    weeks = []
    for week, kickoff in games:
        w = _week(week)
        ko = _utc(kickoff)
        if w is None or pd.isna(ko):
            continue
        if ko > now:
            weeks.append(w)
    return min(weeks) if weeks else None


def is_decidable(week, kickoff_utc, now_utc, decision_week, max_days=MAX_DECISION_DAYS):
    w = _week(week)
    ko = _utc(kickoff_utc)
    now = _utc(now_utc)
    if decision_week is None or w is None or pd.isna(ko):
        return False
    if w != int(decision_week):
        return False
    return now < ko <= now + pd.Timedelta(days=max_days)


def missing_results(schedule_df, now_utc, grace_hours=RESULTS_GRACE_HOURS):
    """game_ids that kicked off more than grace_hours ago and have no final score."""
    if schedule_df is None or schedule_df.empty:
        return []
    now = _utc(now_utc)
    ko = pd.to_datetime(schedule_df["kickoff_utc"], utc=True, errors="coerce")
    no_score = schedule_df["home_score"].isna() | schedule_df["away_score"].isna()
    late = ko < now - pd.Timedelta(hours=grace_hours)
    return schedule_df.loc[late & no_score, "game_id"].astype(str).tolist()


def eastern_day(ts):
    return _utc(ts).tz_convert(EASTERN).normalize()


def previous_game_day(schedule_df, team, kickoff_utc, season=None):
    """Eastern calendar day of the team's latest scheduled game (played or not) before kickoff_utc.

    Same season only: the season of the game at kickoff_utc, or `season` if given.
    None if there is no such game.
    """
    if schedule_df is None or schedule_df.empty:
        return None
    team = str(team).strip().upper()
    ko = _utc(kickoff_utc)
    s = schedule_df
    plays = (s["home_team"].astype(str).str.upper() == team) | (s["away_team"].astype(str).str.upper() == team)
    games = s[plays]
    if games.empty:
        return None
    g_ko = pd.to_datetime(games["kickoff_utc"], utc=True, errors="coerce")

    if season is None:
        this_game = games[g_ko == ko]
        if not this_game.empty:
            season = int(this_game["season"].iloc[0])
    if season is not None:
        same = pd.to_numeric(games["season"], errors="coerce") == int(season)
        games, g_ko = games[same], g_ko[same]

    prior = g_ko[g_ko < ko]
    if prior.empty:
        return None
    return eastern_day(prior.max())


def rest_days(prev_day, kickoff_utc):
    """Integer Eastern calendar days from prev_day to the kickoff's Eastern day; None if no previous game."""
    if prev_day is None:
        return None
    return int((eastern_day(kickoff_utc) - prev_day).days)


def capped_rest(schedule_df, team, kickoff_utc, season, cap=21, default=7):
    """v3/v4 training convention: min(days, cap), default when there is no previous game.

    Training carries the previous game across seasons; with the 21-day cap that is the
    same as the default in practice, so a same-season lookup is equivalent.
    """
    days = rest_days(previous_game_day(schedule_df, team, kickoff_utc, season=season), kickoff_utc)
    return default if days is None else min(days, cap)


def sheet_decision_week(sheet_df, now_utc):
    if sheet_df is None or sheet_df.empty or "week" not in sheet_df.columns:
        return None
    return current_decision_week(zip(sheet_df["week"], sheet_df["kickoff_time"]), now_utc)


def row_is_decidable(row, now_utc, decision_week):
    return is_decidable(row.get("week"), row.get("kickoff_time"), now_utc, decision_week)


def decision_guard(sheet_df, schedule_df, now_utc):
    """(decision_week, missing_result_game_ids) for a run."""
    return sheet_decision_week(sheet_df, now_utc), missing_results(schedule_df, now_utc)


def print_missing_results(missing):
    print(
        "Results not in yet for " + str(len(missing)) + " game(s): " + ", ".join(missing)
        + ". Capturing and picking nothing this run; a later run will retry."
    )
