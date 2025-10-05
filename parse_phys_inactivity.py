# build_phys_inactivity.py
import os, requests, pandas as pd

def fetch_all(code, top=1000):
    rows, skip = [], 0
    while True:
        r = requests.get(
            f"https://ghoapi.azureedge.net/api/{code}",
            params={"$format":"json", "$top": top, "$skip": skip},
            timeout=60
        )
        r.raise_for_status()
        batch = r.json().get("value", [])
        if not batch: break
        rows.extend(batch); skip += top
    return pd.DataFrame(rows)

df = fetch_all("NCD_PAA")  # adults 18+, age-standardized

mask = (
    (df["SpatialDimType"] == "COUNTRY") &
    (df["TimeDim"].between(2010, 2019)) &
    (
        df["Dim1Type"].isna() |
        ((df["Dim1Type"] == "SEX") & (df["Dim1"] == "SEX_BTSX"))  # both sexes
    ) &
    (
        df["Dim2Type"].isna() |
        ((df["Dim2Type"] == "AGEGROUP") & (df["Dim2"] == "AGEGROUP_YEARS18-PLUS"))
    )
)

dff = (
    df.loc[mask, ["SpatialDim", "TimeDim", "NumericValue"]]
      .rename(columns={"SpatialDim":"iso3", "TimeDim":"year", "NumericValue":"phys_inactive_pct"})
)

dff = dff.groupby(["iso3", "year"], as_index=False)["phys_inactive_pct"].mean()
dff["phys_inactive_pct"] = dff["phys_inactive_pct"].round(1)

os.makedirs("data/raw", exist_ok=True)
out = "data/raw/phys_inactivity.csv"
dff.to_csv(out, index=False)

print("Shape:", dff.shape)
print(dff.head())
print("Saved ->", out)
