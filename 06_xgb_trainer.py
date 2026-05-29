import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
import shap

# =========================
# CONFIG SECTION
# =========================

TRAIN_CSV = # Input path of file 05_C22_train_features.csv
VAL_CSV   = # Input path of file 05_C22_val_features.csv
TEST_CSV  = # Input path of file 05_C22_test_features.csv
# TRAIN_CSV = # Input path of file 05_CCS_train_features.csv
# VAL_CSV   = # Input path of file 05_CCS_val_features.csv
# TEST_CSV  = # Input path of file 05_CCS_test_features.csv

OUT_DIR = # Output directory path for 06_xgb_outputs
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 37
np.random.seed(SEED)

N_TRIALS = 200

# Hyperparameter search space
param_grid = {
    "n_estimators": [200, 300, 500, 600, 700, 800],
    "max_depth": [3, 5, 7, 9, 12, 16],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "gamma": [0.0, 0.1, 1.0],
    "reg_alpha": [0.0, 1e-3, 1e-2, 0.1],
    "reg_lambda": [0.1, 1.0, 10.0],
}

# 1) Load CSVs
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# 2) Determine feature columns (same reserved set)
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

X_train = df_train[feature_cols].astype(float).values
X_train = np.nan_to_num(X_train, copy=True, nan=0.0)
y_train = df_train["age"].astype(float).values

X_val = df_val[feature_cols].astype(float).values
X_val = np.nan_to_num(X_val, copy=True, nan=0.0)
y_val = df_val["age"].astype(float).values

X_test = df_test[feature_cols].astype(float).values
X_test = np.nan_to_num(X_test, copy=True, nan=0.0)
y_test = df_test["age"].astype(float).values


# =========================
# OPTUNA SEARCH
# =========================

def objective(trial):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", param_grid["n_estimators"]),
        "max_depth": trial.suggest_categorical("max_depth", param_grid["max_depth"]),
        "learning_rate": trial.suggest_categorical("learning_rate", param_grid["learning_rate"]),
        "subsample": trial.suggest_categorical("subsample", param_grid["subsample"]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", param_grid["colsample_bytree"]),
        "gamma": trial.suggest_categorical("gamma", param_grid["gamma"]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", param_grid["reg_alpha"]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", param_grid["reg_lambda"]),
        "objective": "reg:squarederror",
        "random_state": SEED,
        "verbosity": 0,
        "tree_method": "hist",
        "n_jobs": 1,  # avoid nested parallelism during CV
    }

    model = XGBRegressor(**params)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mean_mae = -np.mean(cv_scores)
    return mean_mae  # minimize MAE


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)

print("Starting Optuna search")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best XGB params:", study.best_params)
print("Best CV MAE:", study.best_value)

# Refit final model on the full training set with best params
best_params = study.best_params.copy()
best_params.update({
    "objective": "reg:squarederror",
    "random_state": SEED,
    "verbosity": 0,
    "tree_method": "hist",
    "n_jobs": -1,
})

XGB = XGBRegressor(**best_params)
XGB.fit(X_train, y_train)

# Save best hyperparams
bp = study.best_params.copy()
bp["best_score"] = float(study.best_value)
pd.DataFrame([bp]).to_csv(os.path.join(OUT_DIR, "xgb_best_params.csv"), index=False)

# Save trained model
model_path = os.path.join(OUT_DIR, "xgb_model.joblib")
joblib.dump(XGB, model_path)

# Plotting Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    estimator=XGB,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    shuffle=True,
    random_state=SEED
)

train_mae = -np.mean(train_scores, axis=1)
val_mae = -np.mean(val_scores, axis=1)

loss_plot = os.path.join(OUT_DIR, "loss.png")

plt.figure()
plt.plot(train_sizes, train_mae, label='Training MAE')
plt.plot(train_sizes, val_mae, label='Validation MAE')
plt.xlabel('Number of training samples')
plt.ylabel('MAE')
plt.title('Learning Curve (XGB)')
plt.legend()
plt.savefig(loss_plot)
plt.show()

# 5) Predictions on train, val, test
BA_tr = XGB.predict(X_train)
BA_va = XGB.predict(X_val)
BA_te = XGB.predict(X_test)

preds_train = pd.DataFrame({'age': y_train, 'BA': BA_tr, 'BAI': BA_tr - y_train})
preds_val   = pd.DataFrame({'age': y_val,   'BA': BA_va, 'BAI': BA_va - y_val})
preds_test  = pd.DataFrame({'age': y_test,   'BA': BA_te, 'BAI': BA_te - y_test})

preds_train.to_csv(os.path.join(OUT_DIR, "preds_train.csv"), index=False)
preds_val.to_csv(os.path.join(OUT_DIR, "preds_val.csv"), index=False)
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)

# 6) Bias correction on test set (same 10-year sliding approach)
starts = np.arange(18, 90, 10)
rows = []
for s in starts:
    mask = (y_train >= s) & (y_train < s + 10)
    if np.sum(mask) == 0:
        bias_train = np.nan
    else:
        bias_train = np.mean(y_train[mask] - BA_tr[mask])
    rows.append({'CA_min': int(s), 'CA_max': int(s + 10), 'bias': bias_train})
bias_df = pd.DataFrame(rows)

# Apply bias correction to test set
bias_for_samples = np.full_like(y_test, fill_value=np.nan, dtype=float)
for _, row in bias_df.iterrows():
    mask = (y_test >= row['CA_min']) & (y_test <= row['CA_max'])
    bias_for_samples[mask] = row['bias']
bias_for_samples[np.isnan(bias_for_samples)] = 0.0

BA_te_corrected = BA_te + bias_for_samples
BAI_te_corrected = BA_te_corrected - y_test

preds_test_corrected = pd.DataFrame(
    {
        "age": y_test,
        "BA": BA_te,
        "BA_corrected": BA_te_corrected,
        "BAI_corrected": BAI_te_corrected,
    }
)
preds_test_corrected.to_csv(
    os.path.join(OUT_DIR, "preds_test_after_bias.csv"), index=False
)

# 7) Compute performance metrics
def safe_metrics(df, pred_col='BA', true_col='age'):
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2_skl = r2_score(y_true, y_pred)

    return mae, rmse, r2_skl

mae_tr, rmse_tr, r2_skl_tr = safe_metrics(preds_train)
mae_va, rmse_va, r2_skl_va = safe_metrics(preds_val)
mae_te_before, rmse_te_before, r2_skl_before = safe_metrics(preds_test)

preds_test_corrected = pd.DataFrame({'age': y_test, 'BA_corrected': BA_te_corrected})
mae_te_after, rmse_te_after, r2_skl_after = safe_metrics(preds_test_corrected, pred_col='BA_corrected')

metrics = {
    'mae_train': mae_tr, 'rmse_train': rmse_tr, 'r2_skl_train': r2_skl_tr,
    'mae_val': mae_va,   'rmse_val': rmse_va,   'r2_skl_val': r2_skl_va,
    'mae_test_before_correction': mae_te_before,
    'rmse_test_before_correction': rmse_te_before,
    'r^2_skl_test_before_correction': r2_skl_before,
    'mae_test_after_correction': mae_te_after,
    'rmse_test_after_correction': rmse_te_after,
    'r^2_skl_test_after_correction': r2_skl_after
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(OUT_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_path, index=False)

# Save preds with corrected BA on test as well
preds_test['BA_corrected'] = BA_te_corrected
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test_with_correction.csv"), index=False)

print("\n[Performance Summary]")
print(metrics_df.T.to_string(header=False))
print(f"\nSaved artifacts to: {OUT_DIR}")


# =========================
# TREE SHAP
# =========================
explainer = shap.TreeExplainer(
    XGB,
    data=X_train,                  # background for interventional E[f(x)]
    feature_perturbation="interventional"  # handles correlated EEG features correctly
)

shap = explainer(X_test, check_additivity=True) # If check_ddditivty is slow, then remove it. Else run it on a samller subset

# --- Save raw SHAP values ---
shap_cols = [f"shap_{c}" for c in feature_cols]
pd.DataFrame(shap.values, columns=shap_cols).to_csv(
    os.path.join(OUT_DIR, "shap_xgb.csv"), index=False
)

# --- Plot 1: Beeswarm (test set) ---
plt.figure()
shap.plots.beeswarm(shap, max_display=20, show=False)
plt.title("TreeSHAP — Beeswarm (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_xgb.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: Bar chart — mean |SHAP| (test set) ---
plt.figure()
shap.plots.bar(shap, max_display=20, show=False)
plt.title("TreeSHAP — Mean |SHAP| (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_xgb.png"), dpi=150, bbox_inches='tight')
plt.show()


# --- Summary table ---
mean_abs_shap = np.abs(shap.values).mean(axis=0)

shap_summary = pd.DataFrame({
    'feature':       feature_cols,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap':     shap.values.mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_summary.to_csv(os.path.join(OUT_DIR, "shap_summary_xgb.csv"), index=False)
print("\n[Top 10 Features by Mean |SHAP| — Test Set]")
print(shap_summary.head(10).to_string(index=False))
print(f"\nSHAP artifacts saved to: {OUT_DIR}")
