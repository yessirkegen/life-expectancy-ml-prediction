# build_tobacco.py
import os, requests, pandas as pd

def fetch_all(code, top=1000):
    rows, skip = [], 0
    while True:
        r = requests.get(f"https://ghoapi.azureedge.net/api/{code}",
                         params={"$format":"json", "$top": top, "$skip": skip},
                         timeout=60)
        r.raise_for_status()
        batch = r.json().get("value", [])
        if not batch: break
        rows.extend(batch); skip += top
    return pd.DataFrame(rows)

CODE = "M_Est_tob_curr_std"  # age-standardized prevalence of current tobacco use (15+)

df = fetch_all(CODE)

# фильтры: страны, 2010–2019
mask = (df["SpatialDimType"] == "COUNTRY") & df["TimeDim"].between(2010, 2019)

# оба пола
if {"Dim1Type","Dim1"}.issubset(df.columns):
    mask &= (df["Dim1Type"].isna()) | ((df["Dim1Type"] == "SEX") & (df["Dim1"].str.contains("BTSX", na=False)))

# возраст 15+
if {"Dim2Type","Dim2"}.issubset(df.columns):
    mask &= (df["Dim2Type"].isna()) | ((df["Dim2Type"] == "AGEGROUP") & (df["Dim2"].str.contains("15", na=False)))

dff = (
    df.loc[mask, ["SpatialDim","TimeDim","NumericValue"]]
      .rename(columns={"SpatialDim":"iso3","TimeDim":"year","NumericValue":"tobacco_pct"})
      .groupby(["iso3","year"], as_index=False)["tobacco_pct"].mean()
)

dff["tobacco_pct"] = dff["tobacco_pct"].astype(float).round(1)

os.makedirs("data/raw", exist_ok=True)
out = "data/raw/tobacco.csv"
dff.to_csv(out, index=False)
print("Shape:", dff.shape)
print(dff.head())
print("Saved ->", out)
