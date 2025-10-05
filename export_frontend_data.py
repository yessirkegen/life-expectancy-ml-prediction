# export_frontend_data.py
# Produces static JSON for a GitHub Pages demo:
# - site/data/preds_2019.json  (model-only predictions for all scenario combos per ISO3)
# - site/data/countries.json    (ISO3 -> country name, via World Bank)

import os, json, joblib, requests
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

MODEL_PATH = "models/best_rf.joblib"
MASTER = "data/processed/master_interpolated.csv"
OUT_DIR = Path("site/data"); OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]

# exactly like training (needed for unpickle)
def log1p_df(X):
    X = X.copy()
    cols = [c for c in ["pm25_ugm3","gdp_pc_usd"] if c in X.columns]
    if cols:
        X[cols] = np.log1p(X[cols])
    return X

# scenario adjustments in feature space (the "as-is" phenomenon)
def apply_user_adjustments(row, smoke, drink, activity, polluted):
    r = row.copy()
    if smoke == "none":       r["tobacco_pct"] = max(0, r["tobacco_pct"] - 10)
    elif smoke == "light":    r["tobacco_pct"] = r["tobacco_pct"] - 5
    elif smoke == "moderate": r["tobacco_pct"] = r["tobacco_pct"] + 10
    elif smoke == "heavy":    r["tobacco_pct"] = r["tobacco_pct"] + 20

    if drink == "none":       r["alcohol_lpa"] = max(0, r["alcohol_lpa"] - 2)
    elif drink == "rare":     r["alcohol_lpa"] = max(0, r["alcohol_lpa"] - 0.5)
    elif drink == "moderate": r["alcohol_lpa"] = r["alcohol_lpa"] + 1
    elif drink == "heavy":    r["alcohol_lpa"] = r["alcohol_lpa"] + 3

    if activity == "low":     r["phys_inactive_pct"] = r["phys_inactive_pct"] + 15
    elif activity == "mid":   r["phys_inactive_pct"] = r["phys_inactive_pct"] + 5
    elif activity == "high":  r["phys_inactive_pct"] = max(0, r["phys_inactive_pct"] - 10)

    if polluted == "yes":     r["pm25_ugm3"] = r["pm25_ugm3"] + 7
    return r

# 1) load model & data
pack = joblib.load(MODEL_PATH)
pipe = pack["model"]; med = pack["medians"]

df = pd.read_csv(MASTER)
for c in NUM + ["life_expectancy"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df2019 = df[df["year"] == 2019].copy().reset_index(drop=True)
iso3_list = sorted(df2019["iso3"].unique())

# 2) generate scenario grid predictions for 2019
SMOKE = ["none","light","moderate","heavy"]
DRINK = ["none","rare","moderate","heavy"]
ACT   = ["low","mid","high"]
POLL  = ["no","yes"]

def key(smoke, drink, act, poll):
    return f"{smoke}|{drink}|{act}|{poll}"

preds = {}
for iso in iso3_list:
    base_row = df2019[df2019["iso3"] == iso].iloc[0].copy()
    bucket = {"base": None, "scenarios": {}}
    # base = "none|none|mid|no"
    base_adj = apply_user_adjustments(base_row, "none","none","mid","no")
    Xb = base_adj[NUM].to_frame().T.fillna(med).infer_objects(copy=False)
    bucket["base"] = float(pipe.predict(Xb)[0])

    for s, d, a, p in product(SMOKE, DRINK, ACT, POLL):
        row = apply_user_adjustments(base_row, s, d, a, p)
        X = row[NUM].to_frame().T.fillna(med).infer_objects(copy=False)
        pred = float(pipe.predict(X)[0])
        bucket["scenarios"][key(s,d,a,p)] = round(pred, 2)
    preds[iso] = bucket

with open(OUT_DIR / "preds_2019.json", "w", encoding="utf-8") as f:
    json.dump({
        "year": 2019,
        "smoke": SMOKE, "drink": DRINK, "activity": ACT, "polluted": POLL,
        "predictions": preds
    }, f, ensure_ascii=False)

print("Saved -> site/data/preds_2019.json")

# 3) countries list with names (World Bank API), filtered by our ISO3
def fetch_wb_countries():
    out, page = [], 1
    while True:
        r = requests.get("https://api.worldbank.org/v2/country",
                         params={"format":"json","per_page":400,"page":page}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]: break
        out.extend(data[1])
        if page >= data[0].get("pages", 1): break
        page += 1
    return out

wb = fetch_wb_countries()
rows = []
for it in wb:
    iso = it.get("id","")
    if iso in iso3_list and it.get("region",{}).get("value") != "Aggregates":
        rows.append({"iso3": iso, "name": it.get("name","")})

# ensure all present (fallback name = iso)
have = {r["iso3"] for r in rows}
for iso in iso3_list:
    if iso not in have:
        rows.append({"iso3": iso, "name": iso})

rows = sorted(rows, key=lambda x: x["name"])
with open(OUT_DIR / "countries.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
print("Saved -> site/data/countries.json")
