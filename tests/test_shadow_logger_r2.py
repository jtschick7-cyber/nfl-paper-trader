import pytest

import src.log_v2_shadow as v2
import src.log_v3_shadow as v3
import src.log_v4_shadow as v4
from src.config import MIN_EDGE_THRESHOLD


@pytest.mark.parametrize("mod", [v2, v3, v4])
def test_suppress_danger_only_at_or_above_threshold(mod):
    below = MIN_EDGE_THRESHOLD - 0.0047
    assert mod.shadow_decision(below, "SUPPRESS") == "PASS"
    assert mod.shadow_decision(below, "MANUAL_REVIEW") == "PASS"
    assert mod.shadow_decision(below, "PROCEED") == "PASS"
    assert mod.shadow_decision(MIN_EDGE_THRESHOLD, "SUPPRESS") == "SUPPRESS_DANGER"
    assert mod.shadow_decision(0.025, "MANUAL_REVIEW") == "MANUAL_REVIEW"
    assert mod.shadow_decision(0.025, "PROCEED") == "SHADOW_PICK"


@pytest.mark.parametrize("mod", [v3, v4])
def test_high_conviction_tier_kept(mod):
    assert mod.shadow_decision(0.04, "PROCEED") == "HIGH_CONVICTION"


@pytest.mark.parametrize("mod", [v2, v3, v4])
def test_headers_keep_r2_columns_before_r3_columns(mod):
    assert mod.HEADERS[-5:-3] == ["artifact_id", "decision_regime"]
    assert len(set(mod.HEADERS)) == len(mod.HEADERS)


def test_v4_local_csv_gains_new_columns(tmp_path):
    import pandas as pd

    path = tmp_path / "v4.csv"
    old_cols = v4.HEADERS[:-5]  # header before R2 and R3
    pd.DataFrame([["x"] * len(old_cols)], columns=old_cols).to_csv(path, index=False)
    v4.append_local_csv([["y"] * len(v4.HEADERS)], path=str(path))
    v4.append_local_csv([["z"] * len(v4.HEADERS)], path=str(path))
    df = pd.read_csv(path, dtype=str)
    assert list(df.columns) == v4.HEADERS
    assert len(df) == 3
    assert df["artifact_id"].isna().iloc[0] and df["artifact_id"].iloc[2] == "z"
    assert df["raw_lean_side"].isna().iloc[0] and df["raw_lean_side"].iloc[2] == "z"
