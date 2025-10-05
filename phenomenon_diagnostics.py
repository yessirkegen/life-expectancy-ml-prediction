# phenomenon_diagnostics.py
# Diagnose the "counter-intuitive lifestyle → higher life expectancy" phenomenon.
# - correlations & partial correlations (align rows properly)
# - ablation: RandomForest WITHOUT GDP
# - GDP-stratified scatter (tobacco vs life expectancy)
# - simple scenario sanity check for KAZ
#
# Run:  python phenomenon_diagnostics.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA  = "data/processed/master_interpolated.csv"
MODEL = "models/best_rf.joblib"
FIGDIR = "figs"
os.makedirs(FIGDIR, exist_ok=True)

NUM = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]
TARGET = "life_expectancy"

# same transform as in training (needed for unpickling)
def log1p_df(X):
    X = X.copy()
    cols = [c for c in ["pm25_ugm3","gdp_pc_usd"] if c in X.columns]
    if cols:
        X[cols] = np.log1p(X[cols])
    return X

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

# ---------- load & split ----------
df = pd.read_csv(DATA)
for c in NUM + [TARGET]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

train = df[df["year"] <= 2017].copy()
val   = df[df["year"] == 2018].copy()
test  = df[df["year"] == 2019].copy()

train_medians = train[NUM].median(numeric_only=True)

Xtr, ytr = train[NUM].fillna(train_medians), train[TARGET]
Xv,  yv  = val[NUM].fillna(train_medians),   val[TARGET]
Xte, yte = test[NUM].fillna(train_medians),  test[TARGET]

# ---------- 1) correlations ----------
print("\n=== Correlations (train <=2017) ===")
corrs = train[["tobacco_pct","alcohol_lpa","phys_inactive_pct","pm25_ugm3","gdp_pc_usd",TARGET]] \
            .corr(numeric_only=True)
print(corrs[TARGET].sort_values(ascending=False))

# ---------- partial correlation helper (proper alignment) ----------
def partial_corr_df(df_in: pd.DataFrame, x: str, y: str, controls: list[str]) -> tuple[float,int]:
    """Residualize x and y on controls after dropping rows with any NaN; return Pearson r and N."""
    cols = [x, y] + controls
    sub = df_in[cols].dropna().copy()
    if sub.empty or sub.shape[0] < 3:
        return float("nan"), 0
    C = sub[controls].values
    lr = LinearRegression()

    lr.fit(C, sub[x].values)
    x_res = sub[x].values - lr.predict(C)

    lr.fit(C, sub[y].values)
    y_res = sub[y].values - lr.predict(C)

    r = float(np.corrcoef(x_res, y_res)[0, 1])
    return r, sub.shape[0]

pc_tob, n_pc = partial_corr_df(train, "tobacco_pct", TARGET, ["gdp_pc_usd", "year"])
print(f"\nPartial corr( tobacco , life | GDP, year ) = {pc_tob:.3f}   (N={n_pc})")

# ---------- 2) ablation: RF WITHOUT GDP ----------
print("\n=== Ablation: RandomForest WITHOUT gdp_pc_usd ===")
num_no_gdp = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","year"]

prep = Pipeline([
    ("log1p", FunctionTransformer(log1p_df, validate=False)),
    ("scale", StandardScaler())
])

rf_no_gdp = Pipeline([
    ("prep", prep),
    ("rf", RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1))
])

med_no_gdp = train[num_no_gdp].median(numeric_only=True)
Xtr_ng = train[num_no_gdp].fillna(med_no_gdp)
Xv_ng  = val[num_no_gdp].fillna(med_no_gdp)
Xte_ng = test[num_no_gdp].fillna(med_no_gdp)

rf_no_gdp.fit(Xtr_ng, ytr)
pv = rf_no_gdp.predict(Xv_ng)
pt = rf_no_gdp.predict(Xte_ng)

print("VAL17→18 no-GDP  : MAE=%.3f RMSE=%.3f R2=%.3f" % (mean_absolute_error(yv,pv), rmse(yv,pv), r2_score(yv,pv)))
print("TEST 2019 no-GDP : MAE=%.3f RMSE=%.3f R2=%.3f" % (mean_absolute_error(yte,pt), rmse(yte,pt), r2_score(yte,pt)))

# ---------- Compare with saved tuned model WITH GDP ----------
print("\n=== Saved tuned RF WITH GDP (reference) ===")
PACK = joblib.load(MODEL)
pipe = PACK["model"]; med_saved = PACK.get("medians", train_medians)

pv_g = pipe.predict(val[NUM].fillna(med_saved))
pt_g = pipe.predict(test[NUM].fillna(med_saved))
print("VAL17→18 WITH GDP: MAE=%.3f RMSE=%.3f R2=%.3f" % (mean_absolute_error(yv,pv_g), rmse(yv,pv_g), r2_score(yv,pv_g)))
print("TEST 2019 WITH GDP: MAE=%.3f RMSE=%.3f R2=%.3f" % (mean_absolute_error(yte,pt_g), rmse(yte,pt_g), r2_score(yte,pt_g)))

# ---------- 3) GDP-stratified scatter ----------
print("\nSaving GDP-stratified scatter…")
# quartiles; duplicates='drop' safeguards if values tie
q = pd.qcut(test["gdp_pc_usd"].rank(method="first"), q=4, labels=["Q1 (lowest)","Q2","Q3","Q4 (highest)"])
colors = {"Q1 (lowest)":"#1f77b4","Q2":"#ff7f0e","Q3":"#2ca02c","Q4 (highest)":"#d62728"}

plt.figure()
for grp, c in colors.items():
    m = (q == grp)
    plt.scatter(test.loc[m, "tobacco_pct"], test.loc[m, TARGET], alpha=0.7, label=grp, s=30, color=c)
plt.xlabel("Tobacco prevalence (% adults)")
plt.ylabel("Life expectancy (years)")
plt.title("2019: Life expectancy vs Tobacco, colored by GDP quartile")
plt.legend(title="GDP per capita quartile")
plt.tight_layout()
path_scatter = os.path.join(FIGDIR, "scatter_tobacco_vs_le_by_gdp.png")
plt.savefig(path_scatter)
print("Saved ->", path_scatter)

# ---------- 4) Simple scenario sanity check (KAZ 2019) ----------
print("\nScenario sanity check: KAZ 2019 (model-only WITH GDP vs ablated no-GDP)")
def get_row(iso3, year):
    sub = df[(df["iso3"] == iso3) & (df["year"] == year)]
    if sub.empty:
        sub = df[df["iso3"] == iso3].copy()
        sub["d"] = (sub["year"] - year).abs()
        sub = sub.sort_values("d").head(1)
    return sub.iloc[0].copy()

def adjust_smoke(row, level):
    r = row.copy()
    if level == "none":       r["tobacco_pct"] = max(0, r["tobacco_pct"] - 10)
    elif level == "light":    r["tobacco_pct"] = r["tobacco_pct"] - 5
    elif level == "moderate": r["tobacco_pct"] = r["tobacco_pct"] + 10
    elif level == "heavy":    r["tobacco_pct"] = r["tobacco_pct"] + 20
    return r

row = get_row("KAZ", 2019)
for s in ["none","light","moderate","heavy"]:
    r = adjust_smoke(row, s)
    Xg = r[NUM].to_frame().T.fillna(med_saved)
    Xn = r[num_no_gdp].to_frame().T.fillna(med_no_gdp)
    pred_g = float(pipe.predict(Xg)[0])
    pred_n = float(rf_no_gdp.predict(Xn)[0])
    print(f"smoke={s:>8s} -> WITH GDP: {pred_g:6.2f}   no-GDP: {pred_n:6.2f}")

print("\nDone.")
