# build_master.py
import pandas as pd
from pathlib import Path
from functools import reduce

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load(name, value_name):
    df = pd.read_csv(RAW / name, dtype={"iso3":str, "year":int})
    assert {"iso3","year",value_name}.issubset(df.columns), f"Bad cols in {name}: {df.columns}"
    return df[["iso3","year",value_name]]

le   = load("life_expectancy.csv", "life_expectancy")
alc  = load("alcohol.csv",          "alcohol_lpa")
tob  = load("tobacco.csv",          "tobacco_pct")
ipa  = load("phys_inactivity.csv",  "phys_inactive_pct")
pm25 = load("pm25.csv",             "pm25_ugm3")
gdp  = load("gdp_pc.csv",           "gdp_pc_usd")

dfs = [le, alc, tob, ipa, pm25, gdp]
master = reduce(lambda l, r: pd.merge(l, r, on=["iso3","year"], how="inner"), dfs)

# удалим дубли
master = master.groupby(["iso3","year"], as_index=False).mean(numeric_only=True)

# ограничим 2010–2019
master = master.query("2010 <= year <= 2019").sort_values(["iso3","year"]).reset_index(drop=True)

# мини-EDA
print("Shape:", master.shape)
print("\nMissing ratio (top):")
print(master.isna().mean().sort_values(ascending=False).head(10))
print("\nHead:")
print(master.head())

# сохраним
out_path = OUT / "master.csv"
master.to_csv(out_path, index=False)
print("\nSaved ->", out_path)
