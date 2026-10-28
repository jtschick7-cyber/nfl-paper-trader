"""
Unit tests for Odds Utilities, Vig Removal, and Multi-Book Line Shopping.
"""

import pytest
from src.odds_utils import (
    american_to_decimal_odds,
    american_to_implied_prob,
    parse_spreads,
    remove_vig,
)


def test_american_to_implied_prob():
    # Even money
    assert pytest.approx(american_to_implied_prob(100.0), 1e-4) == 0.50
    assert pytest.approx(american_to_implied_prob(-100.0), 1e-4) == 0.50

    # Standard spread -110
    assert pytest.approx(american_to_implied_prob(-110.0), 1e-4) == 110.0 / 210.0

    # Underdog +150
    assert pytest.approx(american_to_implied_prob(150.0), 1e-4) == 100.0 / 250.0

    # Favorite -200
    assert pytest.approx(american_to_implied_prob(-200.0), 1e-4) == 200.0 / 300.0


def test_remove_vig():
    # Symmetric -110 / -110
    imp_h = american_to_implied_prob(-110.0)
    imp_a = american_to_implied_prob(-110.0)
    fair_h, fair_a = remove_vig(imp_h, imp_a)

    assert pytest.approx(fair_h, 1e-6) == 0.50
    assert pytest.approx(fair_a, 1e-6) == 0.50
    assert pytest.approx(fair_h + fair_a, 1e-6) == 1.0

    # Asymmetric -130 / +110
    imp_h = american_to_implied_prob(-130.0)
    imp_a = american_to_implied_prob(110.0)
    fair_h, fair_a = remove_vig(imp_h, imp_a)
    assert pytest.approx(fair_h + fair_a, 1e-6) == 1.0
    assert fair_h > fair_a


def test_american_to_decimal_odds():
    assert pytest.approx(american_to_decimal_odds(-110.0), 1e-4) == 1.0 + 100.0 / 110.0
    assert pytest.approx(american_to_decimal_odds(150.0), 1e-4) == 2.50
    assert pytest.approx(american_to_decimal_odds(100.0), 1e-4) == 2.0


def test_parse_spreads_multi_book():
    mock_game = {
        "home_team": "Green Bay Packers",
        "away_team": "Detroit Lions",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Green Bay Packers", "point": -3.0, "price": -105.0},
                            {"name": "Detroit Lions", "point": 3.0, "price": -115.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 47.5, "price": -110.0},
                            {"name": "Under", "point": 47.5, "price": -110.0},
                        ],
                    },
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Green Bay Packers", "point": -2.5, "price": -115.0},
                            {"name": "Detroit Lions", "point": 2.5, "price": -105.0},
                        ],
                    }
                ],
            },
            {
                "key": "betonlineag",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Green Bay Packers", "point": -3.0, "price": -110.0},
                            {"name": "Detroit Lions", "point": 3.0, "price": -110.0},
                        ],
                    }
                ],
            },
        ],
    }

    parsed = parse_spreads([mock_game])
    assert len(parsed) == 1
    p = parsed.iloc[0]

    assert p["home_team"] == "GB"
    assert p["away_team"] == "DET"
    assert p["books_count"] == 3
    # Consensus fair prob should sum to 1.0
    assert pytest.approx(p["consensus_fair_home"] + p["consensus_fair_away"], 1e-4) == 1.0
    # Best home line: -2.5 (from FanDuel) is better than -3.0
    assert p["best_home_line"] == -2.5
    # Best away line: +3.0 (from DraftKings / BetOnline) is better than +2.5
    assert p["best_away_line"] == 3.0
    assert p["total_line"] == 47.5


def test_get_cached_live_home_spread_and_priority(tmp_path, monkeypatch):
    import json
    from src.odds_utils import get_cached_live_home_spread

    mock_cache = tmp_path / "odds_latest.json"
    mock_data = [
        {
            "home_team": "Green Bay Packers",
            "away_team": "Detroit Lions",
            "bookmakers": [
                {
                    "key": "bovada",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Green Bay Packers", "point": -1.5, "price": -110.0},
                                {"name": "Detroit Lions", "point": 1.5, "price": -110.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Green Bay Packers", "point": -3.0, "price": -105.0},
                                {"name": "Detroit Lions", "point": 3.0, "price": -115.0},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    mock_cache.write_text(json.dumps(mock_data), encoding="utf-8")
    monkeypatch.setattr("src.odds_utils.ODDS_CACHE_FILE", str(mock_cache))

    # DraftKings has higher priority in BOOK_PRIORITY than bovada, so spread should be -3.0
    spread = get_cached_live_home_spread("GB", "DET")
    assert spread == -3.0

    # Also works with full names
    spread_full = get_cached_live_home_spread("Green Bay Packers", "Detroit Lions")
    assert spread_full == -3.0

    # Non-existent game returns None
    assert get_cached_live_home_spread("KC", "BAL") is None


def test_get_cached_consensus_market(tmp_path, monkeypatch):
    import json
    import src.odds_utils as ou

    def book(key, hp, hpr, ap, apr, total=None):
        markets = [{
            "key": "spreads",
            "outcomes": [
                {"name": "Green Bay Packers", "point": hp, "price": hpr},
                {"name": "Detroit Lions", "point": ap, "price": apr},
            ],
        }]
        if total is not None:
            markets.append({"key": "totals", "outcomes": [
                {"name": "Over", "point": total, "price": -110.0},
                {"name": "Under", "point": total, "price": -110.0},
            ]})
        return {"key": key, "markets": markets}

    data = [{
        "home_team": "Green Bay Packers",
        "away_team": "Detroit Lions",
        "bookmakers": [
            book("draftkings", -3.0, -105.0, 3.0, -115.0, total=47.5),
            book("fanduel", -2.5, -115.0, 2.5, -105.0),
            book("betonlineag", -3.0, -110.0, 3.0, -110.0),
        ],
    }]
    cache = tmp_path / "odds_latest.json"
    cache.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ou, "ODDS_CACHE_FILE", str(cache))

    mkt = ou.get_cached_consensus_market("GB", "DET")
    assert mkt is not None
    assert pytest.approx(mkt["consensus_fair_home"] + mkt["consensus_fair_away"], 1e-4) == 1.0
    assert mkt["best_home_line"] == -2.5
    assert mkt["best_home_book"] == "fanduel" or mkt["best_home_book"]
    assert mkt["best_away_line"] == 3.0
    assert mkt["total_line"] == 47.5
    assert ou.get_cached_consensus_market("KC", "DET") is None

    cache.write_text("not json", encoding="utf-8")
    assert ou.get_cached_consensus_market("GB", "DET") is None
