"""Raw lean: which side a model favours before the market blend, and by how much (R3).

    market_mu    = -home_line
    raw_lean_pts = raw_mu_home_margin - market_mu      (> 0 leans HOME)

Never derive the lean from v*_best_side: that column is driven by price and line
number and disagrees with the raw-lean side on most v3/v4 rows.
"""

from src.market_blend import market_mu

RAW_LEAN_HEADERS = ["raw_mu_home_margin", "raw_lean_pts", "raw_lean_side"]
LEAN_DECIMALS = 4


def lean_side(lean_pts):
    """HOME if the lean is positive, AWAY if negative, blank if exactly zero."""
    if lean_pts > 0:
        return "HOME"
    if lean_pts < 0:
        return "AWAY"
    return ""


def raw_lean_cells(raw_mu, home_line, decimals=LEAN_DECIMALS):
    """[raw_mu_home_margin, raw_lean_pts, raw_lean_side] for a shadow row."""
    raw = round(float(raw_mu), decimals)
    pts = round(float(raw_mu) - market_mu(home_line), decimals)
    pts = pts + 0.0  # -0.0 -> 0.0
    return raw, pts, lean_side(pts)


def unblend_margin(blended_mu, home_line, beta):
    """Inverse of market_blend.blend_margin: the raw model margin from a blended one."""
    m = market_mu(home_line)
    return m + (float(blended_mu) - m) / float(beta)
