# train_baselines.py  (версия с импутацией медианой)
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# --- 1) load ---
df = pd.read_csv("data/processed/master_interpolated.csv")

target = "life_expectancy"
num_features = ["alcohol_lpa","tobacco_pct","phys_inactive_pct","pm25_ugm3","gdp_pc_usd","year"]

for c in num_features + [target]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- 2) time-aware split ---
train = df[df["year"] <= 2017].copy()
val   = df[df["year"] == 2018].copy()
test  = df[df["year"] == 2019].copy()

X_train, y_train = train[num_features].copy(), train[target].copy()
X_val,   y_val   = val[num_features].copy(),   val[target].copy()
X_test,  y_test  = test[num_features].copy(),  test[target].copy()

# --- 3) иммутация пропусков медианами, считанными по train ---
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_val   = X_val.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

assert not X_train.isna().any().any(), "NaN остались в X_train"
assert not X_val.isna().any().any(),   "NaN остались в X_val"
assert not X_test.isna().any().any(),  "NaN остались в X_test"

# --- 4) препроцессинг (лог + скейл) ---
log_cols = ["pm25_ugm3","gdp_pc_usd"]
def log1p_df(X):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X

preprocess = Pipeline([
    ("log1p", FunctionTransformer(log1p_df, validate=False)),
    ("scale", StandardScaler()),
])

# --- 5) модели ---
models = {
    "RidgeCV": RidgeCV(alphas=[0.1, 1.0, 3.0, 10.0, 30.0, 100.0]),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42
    ),
    "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance")
}

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # вместо squared=False
    r2   = r2_score(y_true, y_pred)
    return round(mae,3), round(rmse,3), round(r2,3)

rows = []
best_name, best_rmse, best_pipe = None, float("inf"), None

for name, est in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", est)])
    pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    mae, rmse, r2 = eval_metrics(y_val, pred_val)
    rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    if rmse < best_rmse:
        best_rmse, best_name, best_pipe = rmse, name, pipe

res = pd.DataFrame(rows).sort_values("RMSE")
print("Validation (2018):")
print(res.to_string(index=False))

# финальный тест на 2019
pred_test = best_pipe.predict(X_test)
mae, rmse, r2 = eval_metrics(y_test, pred_test)
print(f"\nBest={best_name} → Test(2019): MAE={mae}, RMSE={rmse}, R2={r2}")
