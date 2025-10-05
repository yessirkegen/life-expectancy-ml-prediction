# build_master_interpolated.py
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# 1) База (iso3-year на 2010–2019)
le = pd.read_csv(RAW / "life_expectancy.csv", dtype={"iso3":str, "year":int})
le = le.query("2010 <= year <= 2019")[["iso3","year","life_expectancy"]].copy()

# 2) Остальные фичи
alc  = pd.read_csv(RAW / "alcohol.csv",          dtype={"iso3":str, "year":int})
tob  = pd.read_csv(RAW / "tobacco.csv",          dtype={"iso3":str, "year":int})
ipa  = pd.read_csv(RAW / "phys_inactivity.csv",  dtype={"iso3":str, "year":int})
pm25 = pd.read_csv(RAW / "pm25.csv",             dtype={"iso3":str, "year":int})
gdp  = pd.read_csv(RAW / "gdp_pc.csv",           dtype={"iso3":str, "year":int})

# 3) Левый джойн к сетке LE (чтобы сохранить все годы)
df = le.merge(alc,  on=["iso3","year"], how="left")
df = df.merge(tob,  on=["iso3","year"], how="left")
df = df.merge(ipa,  on=["iso3","year"], how="left")
df = df.merge(pm25, on=["iso3","year"], how="left")
df = df.merge(gdp,  on=["iso3","year"], how="left")

# 4) Интерполяция по каждой стране для числовых фичей (кроме таргета)
num_cols = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd"]
df = df.sort_values(["iso3","year"]).reset_index(drop=True)

def _interp_group(g):
    g = g.sort_values("year")
    for c in num_cols:
        if c in g:
            # линейная интерполяция, затем ffill/bfill, чтобы заполнить края (2016–2019 из 2015)
            g[c] = g[c].interpolate(method="linear", limit_direction="both")
            g[c] = g[c].ffill().bfill()
    return g

df = df.groupby("iso3", as_index=False).apply(_interp_group).reset_index(drop=True)

# 5) Быстрый контроль пропусков
print("Shape:", df.shape)
print("\nMissing ratio:")
print(df[["life_expectancy"] + num_cols].isna().mean().sort_values(ascending=False))

# 6) Сохранить
out_path = OUT / "master_interpolated.csv"
df.to_csv(out_path, index=False)
print("\nSaved ->", out_path)
