# tune_rf.py
import pandas as pd, numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

# --- load & split (с интерполяцией) ---
df = pd.read_csv("data/processed/master_interpolated.csv")
target = "life_expectancy"
num = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]

train = df[df["year"] <= 2017].copy()
val   = df[df["year"] == 2018].copy()
test  = df[df["year"] == 2019].copy()

Xtr, ytr = train[num], train[target]
Xv,  yv  = val[num],   val[target]
Xte, yte = test[num],  test[target]

# safety: numeric & simple impute by train medians
for c in num+[target]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
med = Xtr.median(numeric_only=True)
Xtr = Xtr.fillna(med); Xv = Xv.fillna(med); Xte = Xte.fillna(med)

# --- preprocess ---
def log1p_df(X):
    X = X.copy()
    for c in ["pm25_ugm3","gdp_pc_usd"]:
        X[c] = np.log1p(X[c])
    return X
prep = Pipeline([("log1p", FunctionTransformer(log1p_df, validate=False)),
                 ("scale", StandardScaler())])

base = Pipeline([("prep", prep),
                 ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])

# --- search space ---
param_dist = {
    "rf__n_estimators": randint(400, 1201),
    "rf__max_depth": [None] + list(range(5, 61, 5)),
    "rf__min_samples_split": randint(2, 11),
    "rf__min_samples_leaf": randint(1, 5),
    "rf__max_features": ["sqrt", "log2", 1.0, 0.7, 0.5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=50,
    cv=cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(Xtr, ytr)

def rmse_from_neg_mse(score):
    return float(np.sqrt(-score))

print("\nBest params:", search.best_params_)
print("CV best RMSE:", rmse_from_neg_mse(search.best_score_))

best = search.best_estimator_

# --- eval on VAL ---
pv = best.predict(Xv)
mae = mean_absolute_error(yv, pv)
rmse = float(np.sqrt(mean_squared_error(yv, pv)))
r2 = r2_score(yv, pv)
print("VAL tuned RF -> MAE=%.3f RMSE=%.3f R2=%.3f" % (mae, rmse, r2))

# --- retrain on train+val and test on 2019 ---
Xtrv = pd.concat([Xtr, Xv], axis=0); ytrv = pd.concat([ytr, yv], axis=0)
best.fit(Xtrv, ytrv)
pte = best.predict(Xte)
mae_t = mean_absolute_error(yte, pte)
rmse_t = float(np.sqrt(mean_squared_error(yte, pte)))
r2_t = r2_score(yte, pte)
print("TEST 2019 tuned RF -> MAE=%.3f RMSE=%.3f R2=%.3f" % (mae_t, rmse_t, r2_t))

# --- save model & train medians for later inference ---
import os
os.makedirs("models", exist_ok=True)
joblib.dump({"model": best, "medians": med}, "models/best_rf.joblib")
print("Saved -> models/best_rf.joblib")
