"""
Review repro 2 (FABLE_REVIEW.md W2, M5): re-derives v3's market beta and calibrator slope
in memory from data/team_game_epa.csv and the nflverse schedule cache, plus the raw-lean
ATS record and the share of calibrator input explained by the line alone.
Writes no model files. Takes about a minute.

    .venv/bin/python scripts/review_repro/v3_beta_rederive.py [optional_out.csv]
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from src.model_utils import _load_sched
from src.train_model_v3 import build_v3_training_dataset, fit_constrained_ridge, V3_FEATURES
from src.market_blend import fit_market_beta, blend_margin
from src.key_numbers import calculate_spread_probabilities

t0=time.time()
seasons=list(range(2012,2026))
epa=pd.read_csv("data/team_game_epa.csv")
sched=_load_sched(list(range(2010,2026)))
ds=build_v3_training_dataset(seasons,epa,sched)
print("rows",len(ds),"build secs",round(time.time()-t0))
X=ds[V3_FEATURES].values; y=ds["actual_home_margin"].values
idx=[];mu=[]
for tr,va in TimeSeriesSplit(n_splits=5).split(ds):
    sc=StandardScaler().fit(X[tr]); r=fit_constrained_ridge(sc.transform(X[tr]),y[tr],alpha=15.0)
    idx.extend(va); mu.extend(r.predict(sc.transform(X[va])))
o=ds.iloc[idx].copy(); o["mu"]=mu
beta,se,n=fit_market_beta(o.mu,o.home_line,o.actual_home_margin)
print(f"beta={beta:.4f} se={se:.4f} n={n}  (metadata_v3: 0.0369 / 0.0710 / 3050)")
o["mkt"]=-o.home_line; o["d"]=o.mu-o.mkt; o["r"]=o.actual_home_margin-o.mkt
print("RMSE model",round(np.sqrt(np.mean((o.actual_home_margin-o.mu)**2)),3),"RMSE market",round(np.sqrt(np.mean(o.r**2)),3))
print("raw disagreement |d|: mean",round(o.d.abs().mean(),2),"median",round(o.d.abs().median(),2),"sd",round(o.d.std(),2),"mean d",round(o.d.mean(),3))
# raw-lean ATS record vs the closing line, by lean size
o["cov"]=o.actual_home_margin+o.home_line
nz=o[o["cov"]!=0]
for th in [0,1.5,3,5]:
    s=nz[nz.d.abs()>=th]; w=(np.sign(s.d)==np.sign(s["cov"])).mean()
    print(f"  raw lean ATS vs close, |lean|>={th}: n={len(s)} win%={w*100:.2f} (se {100*(w*(1-w)/len(s))**0.5:.2f})  break-even 52.38")
# beta by era
for a,b in [(2014,2019),(2020,2025)]:
    s=o[o.season.between(a,b)]; bb,ss,nn=fit_market_beta(s.mu,s.home_line,s.actual_home_margin)
    raw=float((s.d*s.r).sum()/(s.d**2).sum()); print(f"  beta {a}-{b}: raw={raw:.4f} se={ss:.4f} n={nn}")
# calibrator slope and its standard error on the blended probabilities
bs=float(np.std(o.actual_home_margin-[blend_margin(m,h,beta) for m,h in zip(o.mu,o.home_line)]))
p=np.array([calculate_spread_probabilities(blend_margin(m,h,beta),bs,h)["cond_home_win"] for m,h in zip(o.mu,o.home_line)])
p0=np.array([calculate_spread_probabilities(-h,bs,h)["cond_home_win"] for h in o.home_line])
keep=(o["cov"]!=0).values
x=np.log(p[keep]/(1-p[keep])); yy=(o["cov"].values[keep]>0).astype(float)
def slope_se(x,yy):
    b=0.0
    for _ in range(50):
        q=1/(1+np.exp(-b*x)); g=np.sum(x*(yy-q)); h=np.sum(x*x*q*(1-q)); b+=g/h
    return b,1/np.sqrt(h)
b,sb=slope_se(x,yy); print(f"calibrator slope (no intercept, home rows) = {b:.3f} +/- {sb:.3f}   (artifact stores 0.361)")
x0=np.log(p0[keep]/(1-p0[keep])); b0,sb0=slope_se(x0,yy); print(f"same fit using ONLY the line (zero model input) = {b0:.3f} +/- {sb0:.3f}")
lp=np.log(p/(1-p)); lp0=np.log(p0/(1-p0))
print("share of logit(p_raw) variance explained by the line alone:",round(np.corrcoef(lp,lp0)[0,1]**2,4))
print("sd of p from line-only anchor:",round(p0.std(),4)," sd of the model's own contribution (p - p_line):",round((p-p0).std(),4))
if len(sys.argv) > 1:
    o.assign(p_blend=p,p_line=p0)[["game_id","season","week","home_line","actual_home_margin","mu","d","p_blend","p_line"]].to_csv(sys.argv[1],index=False)
print("total secs",round(time.time()-t0))
