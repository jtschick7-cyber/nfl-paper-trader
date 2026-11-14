"""
Review repro 1 (FABLE_REVIEW.md W1, F3, M4, M5): what the committed shadow artifacts do
when the model has NO disagreement with the market, and whether history supports the
key-number anchor. Read-only: loads models/*.pkl and data/nflverse_cache/games.csv.

    .venv/bin/python scripts/review_repro/anchor_and_price_checks.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import joblib, numpy as np, pandas as pd
from src.key_numbers import calculate_spread_probabilities
from src.model_v2_core import calibrate_probability
from src.odds_utils import american_to_implied_prob, remove_vig
from src.market_blend import blend_margin

arts = {v: joblib.load(f"models/nfl_spread_model{'' if v=='v1' else '_'+v}.pkl") for v in ["v2","v3","v4"]}
print("== 1. calibrators stored in the .pkl artifacts")
for v,a in arts.items():
    print(v, a.get("calibrator_method"), a.get("calibrator_params"), "beta", a.get("market_beta"), "sigma", round(a["sigma"],3),
          "anchor", a.get("anchor", "line"), a.get("anchor_variant", ""))

print("\n== 2. zero-disagreement repro (raw model margin == market margin) -> decision driven by line/price only")
print("   Runs src.decision.shadow_decision, the rule the loggers run, with each artifact's own fields.")
print("   Price-anchored artifacts (R5): zero disagreement is raw_mu = m* (variant A) or -home_line (B).")
from src.decision import shadow_decision
def decide(v, home_line, ho, ao, raw_dis=0.0):
    a = arts[v]
    kw = dict(sigma=a["sigma"], market_beta=a.get("market_beta"),
              calibrator=a.get("calibrator", (a["calibrator_method"], a["calibrator_params"])),
              anchor=a.get("anchor"), anchor_variant=a.get("anchor_variant"))
    zero = -home_line
    if a.get("anchor") == "price" and a.get("anchor_variant", "A") == "A":
        zero = shadow_decision(zero, home_line, ho, ao, **kw)["anchor_mean"]
    d = shadow_decision(zero + raw_dis, home_line, ho, ao, **kw)
    return d["p_home_discrete"], d["p_home_cal"], d["best_side"], d["best_edge"]
rows=[]
for v in ["v2","v3","v4"]:
    for hl,ho,ao in [(-3,-110,-110),(-3,110,-130),(3,-130,110),(-7.5,-110,-110),(7.5,-110,-110),(-2.5,-110,-110),(3.5,-110,-110),(-3,105,-125),(1.5,112,-135)]:
        pd_, pc, side, e = decide(v,hl,ho,ao)
        rows.append((v,hl,ho,ao,round(pd_,4),round(pc,4),side,round(e,4),"PICK" if e>=0.02 else "PASS"))
print(pd.DataFrame(rows,columns=["model","home_line","home_odds","away_odds","p_home_disc","p_home_cal","best","edge","decision"]).to_string(index=False))

print("\n== 3. how big a RAW disagreement (pts) is needed to reach edge>=2% at -110/-110, home_line=-4.5 (dead number)")
for v in ["v2","v3","v4"]:
    need=None
    for d in np.arange(0,60.25,0.25):
        _,_,_,e = decide(v,-4.5,-110,-110,raw_dis=d)
        if e>=0.02: need=d; break
    print(v, "needs raw model-vs-market disagreement >=", need, "pts")

print("\n== 4. anchor bias: P(home covers | mean = -home_line), sigma=12.81, no model disagreement")
out=[]
for hl in [-10,-7.5,-7,-6.5,-3.5,-3,-2.5,-1.5,1.5,2.5,3,3.5,6.5,7,7.5,10]:
    r=calculate_spread_probabilities(-hl,12.8145,hl)
    out.append((hl,round(r["cond_home_win"],4),round(r["p_push"],4)))
print(pd.DataFrame(out,columns=["home_line","cond_home_win","p_push"]).T.to_string(header=False))

print("\n== 5. history check 2012-2025 REG (nflverse closing spread_line): does the hook-holder cover more than 50%?")
g=pd.read_csv("data/nflverse_cache/games.csv",low_memory=False)
t=g[(g.game_type=="REG")&(g.season.between(2012,2025))&g.home_score.notna()].copy()
t["home_line"]=-t.spread_line; t["margin"]=t.home_score-t.away_score; t["cov"]=t.margin+t.home_line
t=t[t["cov"]!=0].copy(); t["home_cover"]=(t["cov"]>0).astype(int)
t["p_anchor"]=[calculate_spread_probabilities(-hl,12.8145,hl)["cond_home_win"] for hl in t.home_line]
print("games (no push):",len(t)," mean p_anchor:",round(t.p_anchor.mean(),4)," actual home cover:",round(t.home_cover.mean(),4))
for lo,hi,lab in [(0,0.485,"model says home <48.5%"),(0.485,0.515,"48.5-51.5%"),(0.515,1,"model says home >51.5%")]:
    s=t[(t.p_anchor>=lo)&(t.p_anchor<hi)]
    se=(s.home_cover.mean()*(1-s.home_cover.mean())/len(s))**0.5
    print(f"  {lab:26s} n={len(s):4d} predicted={s.p_anchor.mean():.4f} actual={s.home_cover.mean():.4f} (se {se:.4f})")
