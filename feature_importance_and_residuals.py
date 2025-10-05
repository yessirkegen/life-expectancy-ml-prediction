# feature_importance_and_residuals.py
# Analyze a saved RandomForest pipeline: permutation importance + residuals.
# Uses the interpolated dataset and your saved model pack: models/best_rf.joblib

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ---------- IMPORTANT: define EXACTLY the same function used in training ----------
def log1p_df(X):
    """Log1p-transform the skewed columns exactly as during training."""
    X = X.copy()
    if "pm25_ugm3" in X.columns and "gdp_pc_usd" in X.columns:
        X[["pm25_ugm3", "gdp_pc_usd"]] = np.log1p(X[["pm25_ugm3", "gdp_pc_usd"]])
    return X
# ----------------------------------------------------------------------------------

MODEL_PATH = "models/best_rf.joblib"
DATA_PATH  = "data/processed/master_interpolated.csv"
FIG_DIR    = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# Load model pack (pipeline + train medians used for imputation)
PACK = joblib.load(MODEL_PATH)
if isinstance(PACK, dict):
    pipe = PACK["model"]
    med  = PACK.get("medians", None)
else:
    pipe = PACK
    med  = None

# Columns in the pipeline
NUM = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]

# Load data (interpolated master) and build splits (train<=2017, test=2019)
df = pd.read_csv(DATA_PATH)
for c in NUM + ["life_expectancy"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

train = df[df["year"] <= 2017].copy()
test  = df[df["year"] == 2019].copy()

Xtr, ytr = train[NUM].copy(), train["life_expectancy"].copy()
Xte, yte = test[NUM].copy(),  test["life_expectancy"].copy()

# Impute using train medians (same as during training)
if med is None:
    med = Xtr.median(numeric_only=True)
Xtr = Xtr.fillna(med)
Xte = Xte.fillna(med)

# ---------- Base metrics on TEST ----------
pred = pipe.predict(Xte)
base_mse = float(mean_squared_error(yte, pred))
base_rmse = float(np.sqrt(base_mse))
base_mae = float(mean_absolute_error(yte, pred))
base_r2  = float(r2_score(yte, pred))

print("Test metrics (saved model): "
      f"MAE={base_mae:.3f} RMSE={base_rmse:.3f} R2={base_r2:.3f}")

# ---------- Permutation importance (convert to ΔRMSE) ----------
# permutation_importance returns decrease in score (here score = neg MSE),
# i.e., importances_mean = MSE_perm - MSE_base  (non-negative).
pi = permutation_importance(
    pipe, Xte, yte,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_squared_error"
)

rows = []
for i, col in enumerate(NUM):
    delta_mse = float(pi.importances_mean[i])              # >= 0
    delta_rmse = float(np.sqrt(base_mse + delta_mse) - np.sqrt(base_mse))
    rows.append((col, delta_mse, delta_rmse))

imp = (pd.DataFrame(rows, columns=["feature","delta_mse","delta_rmse"])
         .sort_values("delta_rmse", ascending=False)
         .reset_index(drop=True))

print("\nPermutation importance (ΔRMSE increase when permuted):")
print(imp.to_string(index=False))

# ---------- Native RF importances ----------
rf_est = pipe.named_steps.get("rf", None)
if rf_est is not None and hasattr(rf_est, "feature_importances_"):
    # Preprocess once to assert 1:1 mapping
    Xt = pipe.named_steps["prep"].transform(Xte)
    fi = rf_est.feature_importances_
    fi_df = (pd.DataFrame({"feature": NUM, "rf_importance": fi})
               .sort_values("rf_importance", ascending=False))
    print("\nRandomForest feature_importances_:")
    print(fi_df.to_string(index=False))

# ---------- Plots ----------
# 1) True vs Predicted scatter
plt.figure()
plt.scatter(yte, pred, alpha=0.6)
plt.xlabel("True life expectancy")
plt.ylabel("Predicted")
plt.title("Test 2019: True vs Predicted")
lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
plt.plot(lims, lims)
plt.tight_layout()
out_scatter = os.path.join(FIG_DIR, "true_vs_pred.png")
plt.savefig(out_scatter)
print(f"\nSaved plot -> {out_scatter}")

# 2) Permutation importance bar chart (ΔRMSE)
plt.figure()
plt.barh(imp["feature"][::-1], imp["delta_rmse"][::-1])
plt.xlabel("ΔRMSE (years) when feature is permuted")
plt.title("Permutation importance (Test 2019)")
plt.tight_layout()
out_bar = os.path.join(FIG_DIR, "perm_importance_delta_rmse.png")
plt.savefig(out_bar)
print(f"Saved plot -> {out_bar}")
