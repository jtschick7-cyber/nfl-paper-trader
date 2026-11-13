"""
Review repro 3 (FABLE_REVIEW.md M2, M3): when was each v1 pick made? Recomputes v1's
margin from the results available at each 3-hour step between line capture and kickoff
and matches it to the `mu=` value stored in Master_Bets notes. Also prints the rest days
the model would have computed at that moment.

Needs a local dump of the Sheet (not committed; data/sheet_dump/ is gitignored):

    mkdir -p data/sheet_dump
    for t in Master_Bets V2_Shadow V3_Shadow V4_Shadow Model_Scorecard; do \
        .venv/bin/python scripts/read_sheet.py $t > data/sheet_dump/$t.tsv; done
    .venv/bin/python scripts/review_repro/v1_pick_timing.py data/sheet_dump
"""
import sys, re, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import joblib, numpy as np, pandas as pd
from src.model_utils import _team_panel, _add_rolling, DIVISION_MAP
S=sys.argv[1]
mb=pd.read_csv(f"{S}/Master_Bets.tsv",sep="\t",dtype=str,keep_default_na=False)
mb["kickoff_utc"]=pd.to_datetime(mb.kickoff_time,utc=True)
def parse(r):
    m=re.match(r"(\w+) (\d+), (\w+) (\d+)",r)
    return (float(m.group(2)),float(m.group(4))) if m else (np.nan,np.nan)
sc=mb.game_result.map(parse)
mb["home_score"]=[a for a,b in sc]; mb["away_score"]=[b for a,b in sc]
mb["season"]=2026; mb["week"]=mb.week.astype(int)
print("completed games by week in Master_Bets:",mb[mb.home_score.notna()].groupby("week").size().to_dict())
model=joblib.load("models/nfl_spread_model.pkl"); meta=json.load(open("models/metadata.json")); fc=meta["feature_cols"]
def snap(as_of):
    comp=mb[(mb.kickoff_utc<as_of)&mb.home_score.notna()][["game_id","season","week","kickoff_utc","home_team","away_team","home_score","away_score"]].copy()
    rolled=_add_rolling(_team_panel(comp),include_current=True); out={}
    for team,g in rolled.groupby("team"):
        last=g.sort_values("kickoff_utc").iloc[-1]
        if pd.isna(last["roll_margin"]): continue
        out[team]={"pf":last.roll_pts_for,"pa":last.roll_pts_against,"margin":last.roll_margin,"last":last.kickoff_utc}
    return out
def mu_at(row,as_of):
    s=snap(as_of); h,a=row.home_team,row.away_team
    if h not in s or a not in s: return None,None
    day=row.kickoff_utc.tz_convert("America/New_York").normalize()
    hr=(day-s[h]["last"].tz_convert("America/New_York").normalize()).days; ar=(day-s[a]["last"].tz_convert("America/New_York").normalize()).days
    v={"diff_pf":s[h]["pf"]-s[a]["pf"],"diff_pa":s[h]["pa"]-s[a]["pa"],"diff_margin":s[h]["margin"]-s[a]["margin"],"diff_rest":float(hr-ar),"division":float(DIVISION_MAP.get(h)==DIVISION_MAP.get(a))}
    return float(model.predict(np.array([[v[c] for c in fc]]))[0]),(hr,ar)
picks=mb[(mb.pick_team!="")&(mb.week>=8)]
rows=[]
for _,r in picks.iterrows():
    m=re.search(r"mu=(-?[\d.]+)",r.notes); noted=float(m.group(1))
    cap=pd.Timestamp(r.opening_captured_at).tz_convert("UTC")
    grid=pd.date_range(cap.floor("h"),min(r.kickoff_utc,pd.Timestamp.now(tz="UTC")),freq="3h")
    hits=[t for t in grid if (lambda x: x is not None and abs(x-noted)<=0.015)(mu_at(r,t)[0])]
    prior_wk=mb[(mb.week==r.week-1)&mb.home_score.notna()]
    n_prior_done=lambda t: int((prior_wk.kickoff_utc<t).sum())
    if hits:
        rows.append((r.game_id[5:],str(cap)[5:16],str(hits[0])[5:16],str(hits[-1])[5:16],n_prior_done(hits[0]),len(prior_wk),round((hits[0]-cap).total_seconds()/86400,1),r.pick_result, r.clv_points))
    else:
        rows.append((r.game_id[5:],str(cap)[5:16],"?","?",None,len(prior_wk),None,r.pick_result,r.clv_points))
df=pd.DataFrame(rows,columns=["game","line_captured","earliest_match","latest_match","prior_wk_games_already_played","prior_wk_games","days_line_was_stale","result","clv_pts"])
pd.set_option("display.width",250); print(df.to_string(index=False))
d=df[df.prior_wk_games_already_played.notna()].copy()
d["la"]=d.prior_wk_games_already_played>0
print("\npicks whose earliest match is after prior-week games were played, by week:")
print(d.groupby(d.game.str[:2]).la.agg(["sum","count"]))
