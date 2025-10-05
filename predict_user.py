# predict_user.py
# Personalized life expectancy prediction using your tuned RF pipeline.
# Uses master_interpolated.csv and models/best_rf.joblib
# Example:
#   python predict_user.py --iso3 KAZ --year 2019 --smoke moderate --drink rare --activity high --polluted no

import argparse, os, sys, joblib, numpy as np, pandas as pd

# IMPORTANT: same function name/signature as in training so joblib can unpickle the pipeline
def log1p_df(X):
    X = X.copy()
    cols = [c for c in ["pm25_ugm3", "gdp_pc_usd"] if c in X.columns]
    if cols:
        X[cols] = np.log1p(X[cols])
    return X

MODEL_PATH = "models/best_rf.joblib"
MASTER_CSV = "data/processed/master_interpolated.csv"
NUM_COLS   = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]

# --- scenario adjustments (simple, transparent heuristics) ---
def apply_user_adjustments(row, smoke, drink, activity, polluted):
    r = row.copy()

    # Tobacco (population %): proxy shift for personal smoking
    if smoke == "none":       r["tobacco_pct"] = max(0, r["tobacco_pct"] - 10)
    elif smoke == "light":    r["tobacco_pct"] = r["tobacco_pct"] - 5
    elif smoke == "moderate": r["tobacco_pct"] = r["tobacco_pct"] + 10
    elif smoke == "heavy":    r["tobacco_pct"] = r["tobacco_pct"] + 20

    # Alcohol (liters per adult/year)
    if drink == "none":       r["alcohol_lpa"] = max(0, r["alcohol_lpa"] - 2)
    elif drink == "rare":     r["alcohol_lpa"] = max(0, r["alcohol_lpa"] - 0.5)
    elif drink == "moderate": r["alcohol_lpa"] = r["alcohol_lpa"] + 1
    elif drink == "heavy":    r["alcohol_lpa"] = r["alcohol_lpa"] + 3

    # Physical inactivity (% of adults): invert based on activity
    if activity == "low":     r["phys_inactive_pct"] = r["phys_inactive_pct"] + 15
    elif activity == "mid":   r["phys_inactive_pct"] = r["phys_inactive_pct"] + 5
    elif activity == "high":  r["phys_inactive_pct"] = max(0, r["phys_inactive_pct"] - 10)

    # Extra PM2.5 exposure (e.g., large polluted city)
    if polluted == "yes":     r["pm25_ugm3"] = r["pm25_ugm3"] + 7

    return r

def load_country_year_row(iso3, year, csv=MASTER_CSV):
    df = pd.read_csv(csv)
    df[NUM_COLS + ["life_expectancy"]] = df[NUM_COLS + ["life_expectancy"]].apply(pd.to_numeric, errors="coerce")
    # exact year match first
    mask = (df["iso3"].str.upper() == iso3.upper()) & (df["year"] == year)
    if mask.any():
        return df.loc[mask].iloc[0]
    # fallback: nearest available year for that ISO3
    sub = df[df["iso3"].str.upper() == iso3.upper()].copy()
    if sub.empty:
        raise SystemExit(f"No data for ISO3={iso3}. Check the code (e.g., KAZ, USA, JPN).")
    sub["dist"] = (sub["year"] - year).abs()
    return sub.sort_values("dist").iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso3", required=True, help="ISO3 country code, e.g., KAZ, USA, JPN")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--smoke", choices=["none","light","moderate","heavy"], default="none")
    ap.add_argument("--drink", choices=["none","rare","moderate","heavy"], default="none")
    ap.add_argument("--activity", choices=["low","mid","high"], default="mid")
    ap.add_argument("--polluted", choices=["no","yes"], default="no")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}")

    pack = joblib.load(MODEL_PATH)      # contains {"model": pipeline, "medians": Series}
    pipe = pack["model"]
    med  = pack.get("medians", None)

    base_row = load_country_year_row(args.iso3, args.year)
    adj_row  = apply_user_adjustments(base_row, args.smoke, args.drink, args.activity, args.polluted)

    X = adj_row[NUM_COLS].to_frame().T
    if med is None:
        med = X.median(numeric_only=True)  # safety
    X = X.fillna(med)

    pred = float(pipe.predict(X)[0])

    print("\n--- Personalized Life Expectancy (years) ---")
    print(f"Country={args.iso3.upper()}  Year={int(adj_row['year'])}")
    print(f"Predicted life expectancy ≈ {pred:.2f} years\n")

    # context dump (what went into the model after scenario shifts)
    show = adj_row[NUM_COLS]
    print("--- Features after scenario adjustments ---")
    for k in NUM_COLS:
        print(f"{k:>20s}: {show[k]:.4f}" if isinstance(show[k], (int,float,np.floating)) else f"{k:>20s}: {show[k]}")
    print("\nAssumptions:",
          f"smoke={args.smoke}, drink={args.drink}, activity={args.activity}, polluted={args.polluted}")
    print("\nNote: Lifestyle adjustments are simple scenario heuristics for personalization, "
          "not causal medical effects.")

if __name__ == "__main__":
    main()
