"""One-off: fill raw_mu_home_margin, raw_lean_pts, raw_lean_side on existing shadow rows (R3).

Fills the three columns only where they are blank:

- V2_Shadow: raw = mu_home_margin (v2 has no market blend).
- V3_Shadow: raw = market_mu + (mu_home_margin - market_mu) / beta. Rows logged with the
  pre-fix artifact (sigma 13.2147, features hardcoded to zero) are skipped.
- V4_Shadow: same with v4's beta. mu is logged to 2 decimals (about +/-0.12 pts), so
  raw_lean_side stays blank when the logged deviation from the market is under 0.01.

beta is read from the current artifact by artifact_id when the row carries one; rows that
predate R2's stamp are matched to the current artifact by sigma. A row whose artifact_id is
not a current artifact is skipped.

All backfilled rows are legacy regime: they inform, but never count toward the raw-lean test.

Dry-run by default; writes only with --apply. Run it inside the sheet-writer concurrency
group (the "Backfill raw leans" workflow).
"""

import argparse
import math

import pandas as pd

from src.raw_lean import RAW_LEAN_HEADERS, raw_lean_cells, unblend_margin

# Constants for rows that predate R2's artifact_id stamp (current artifacts as of 2026-11-02)
V3_BETA = 0.03693757162735288
V4_BETA = 0.042324548578395654
V3_CURRENT_SIGMA = 12.8145
V3_PRE_FIX_SIGMA = 13.2147
V4_CURRENT_SIGMA = 12.81
V4_AMBIGUOUS_DEVIATION = 0.01

TABS = {
    "V2_Shadow": ("v2", "models/nfl_spread_model_v2.pkl"),
    "V3_Shadow": ("v3", "models/nfl_spread_model_v3.pkl"),
    "V4_Shadow": ("v4", "models/nfl_spread_model_v4.pkl"),
}


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() in {"", "nan"}


def _sigma_matches(value, target, decimals):
    s = _num(value)
    return s is not None and round(s, decimals) == round(target, decimals)


def _beta_for(tab, row, betas):
    """beta for a blended row, or None to skip it."""
    aid = row.get("artifact_id")
    if not _blank(aid):
        return betas.get(str(aid).strip())
    if tab == "V3_Shadow":
        return V3_BETA if _sigma_matches(row.get("sigma"), V3_CURRENT_SIGMA, 4) else None
    if tab == "V4_Shadow":
        return V4_BETA if _sigma_matches(row.get("sigma"), V4_CURRENT_SIGMA, 2) else None
    return None


def backfill_values(tab, row, betas=None):
    """(raw_mu_home_margin, raw_lean_pts, raw_lean_side) for one logged row, or None to skip."""
    betas = betas or {}
    mu = _num(row.get("mu_home_margin"))
    home_line = _num(row.get("home_line"))
    if mu is None or home_line is None:
        return None

    if tab == "V2_Shadow":
        return raw_lean_cells(mu, home_line)

    beta = _beta_for(tab, row, betas)
    if not beta:
        return None
    raw, pts, side = raw_lean_cells(unblend_margin(mu, home_line, beta), home_line)
    if tab == "V4_Shadow" and abs(mu + home_line) < V4_AMBIGUOUS_DEVIATION - 1e-9:
        side = ""
    return raw, pts, side


def plan_backfill(tab, df, col, betas=None):
    """(updates, summary). Pure. `col` maps header -> 1-based column and includes RAW_LEAN_HEADERS."""
    updates = []
    summary = {"filled": 0, "skipped": 0, "already_set": 0, "strong": 0, "side_blank": 0, "by_week": {}}
    for idx, row in df.iterrows():
        if all(not _blank(row.get(h)) for h in RAW_LEAN_HEADERS):
            summary["already_set"] += 1
            continue
        vals = backfill_values(tab, row, betas)
        if vals is None:
            summary["skipped"] += 1
            continue
        r = idx + 2
        for h, v in zip(RAW_LEAN_HEADERS, vals):
            if _blank(row.get(h)):
                updates.append({"row": r, "col": col[h], "value": v})
        summary["filled"] += 1
        if abs(vals[1]) >= 1.5:
            summary["strong"] += 1
        if vals[2] == "":
            summary["side_blank"] += 1
        week = str(row.get("week", "")).strip()
        summary["by_week"][week] = summary["by_week"].get(week, 0) + 1
    return updates, summary


def current_betas():
    """{artifact_id: market_beta} for the artifacts in models/."""
    import joblib

    from src.artifacts import artifact_id

    out = {}
    for prefix, path in TABS.values():
        artifact = joblib.load(path)
        beta = artifact.get("market_beta") if isinstance(artifact, dict) else None
        if beta:
            out[artifact_id(path, artifact, prefix)] = float(beta)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write to the Sheet (default: dry-run)")
    args = parser.parse_args()

    import src.log_v2_shadow as v2
    import src.log_v3_shadow as v3
    import src.log_v4_shadow as v4
    from src.sheet_utils import batch_update_cells, ensure_tab_headers, get_or_create_worksheet, retry_api_call

    logger_headers = {"V2_Shadow": v2.HEADERS, "V3_Shadow": v3.HEADERS, "V4_Shadow": v4.HEADERS}
    betas = current_betas()
    print("Current artifacts: " + str(betas))

    for tab, headers in logger_headers.items():
        ws = get_or_create_worksheet(tab)
        if args.apply:
            ensure_tab_headers(ws, headers)
        values = retry_api_call(ws.get_all_values)
        if not values:
            print(tab + ": empty")
            continue
        df = pd.DataFrame(values[1:], columns=values[0])
        col = {h: i + 1 for i, h in enumerate(values[0])}
        for i, h in enumerate(headers):
            col.setdefault(h, i + 1)  # dry run: columns not yet created land at their HEADERS slot
        updates, summary = plan_backfill(tab, df, col, betas)
        print(tab + ": " + str(summary) + ", " + str(len(updates)) + " cell update(s)")
        if args.apply:
            batch_update_cells(ws, updates)
            print(tab + ": applied")

    if not args.apply:
        print("Dry run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
