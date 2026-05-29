import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
import shap

# =========================
# CONFIG SECTION
# =========================

# Dataset filepaths

TRAIN_CSV = # Input path of file 05_C22_train_features.csv
VAL_CSV   = # Input path of file 05_C22_val_features.csv
TEST_CSV  = # Input path of file 05_C22_test_features.csv
# TRAIN_CSV = # Input path of file 05_CCS_train_features.csv
# VAL_CSV   = # Input path of file 05_CCS_val_features.csv
# TEST_CSV  = # Input path of file 05_CCS_test_features.csv

OUT_DIR = # Output directory path for 06_dt_outputs
os.makedirs(OUT_DIR, exist_ok=True)

# Random seed
SEED = 37
np.random.seed(SEED)

# Number of Optuna trials
N_TRIALS = 200

# Search param
param_space = {
    "max_depth": [8, 12, 16, 20, 24, 28, 32, None],
    "min_samples_leaf": [1, 5, 10, 20, 50, 100],
    "min_samples_split": [2, 5, 10, 20, 50, 100, 150,200],
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.01, 0.05],
    "random_state": SEED}

# 1) Load CSVs
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# 2) Determine feature columns (same reserved set)
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

X_train = df_train[feature_cols].astype(float).values
y_train = df_train['age'].astype(float).values

X_val = df_val[feature_cols].astype(float).values
y_val = df_val['age'].astype(float).values

X_test = df_test[feature_cols].astype(float).values
y_test = df_test['age'].astype(float).values

# =============================
# OPTUNA HYPERPARAMETER SEARCH
# =============================

def objective(trial):
    params = {
        "max_depth": trial.suggest_categorical("max_depth",param_space['max_depth']),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", param_space['min_samples_leaf']),
        "min_samples_split": trial.suggest_categorical("min_samples_split", param_space['min_samples_split']),
        "ccp_alpha": trial.suggest_categorical("ccp_alpha", param_space['ccp_alpha']),
        "random_state": SEED,
        "n_jobs" : 1
    }

    model = tree.DecisionTreeRegressor(**params)

    # 5-fold CV on training data only
    # sklearn returns NEGATIVE MAE for this scoring string, so we negate it back.
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mean_mae = -np.mean(cv_scores)
    return mean_mae  # Optuna will minimize this


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best trial:")
print("  MAE:", study.best_value)
print("  Params:", study.best_params)

# Rebuild final model with best parameters and fit on full training set
best_params = study.best_params.copy()
best_params["random_state"] = SEED
DT = tree.DecisionTreeRegressor(**best_params)
DT.fit(X_train, y_train)

# Save best hyperparams
bp = study.best_params.copy()
bp["best_score_mae"] = float(study.best_value)
pd.DataFrame([bp]).to_csv(os.path.join(OUT_DIR, "dt_best_params.csv"), index=False)

# Save trained model
model_path = os.path.join(OUT_DIR, "dt_model.joblib")
joblib.dump(DT, model_path)


# Plotting Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    estimator=DT,
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
plt.title('Learning Curve (DT)')
plt.legend()
plt.savefig(loss_plot)
plt.show()

# 5) Predictions on train, val, test
BA_tr = DT.predict(X_train)
BA_va = DT.predict(X_val)
BA_te = DT.predict(X_test)

preds_train = pd.DataFrame({'age': y_train, 'BA': BA_tr, 'BAI': BA_tr - y_train})
preds_val   = pd.DataFrame({'age': y_val,   'BA': BA_va, 'BAI': BA_va - y_val})
preds_test  = pd.DataFrame({'age': y_test,  'BA': BA_te, 'BAI': BA_te - y_test})

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
    mse = mean_squared_error(y_true, y_pred)  # sklearn MSE
    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2

mae_tr, rmse_tr, r2_tr = safe_metrics(preds_train)
mae_va, rmse_va, r2_va = safe_metrics(preds_val)
mae_te_before, rmse_te_before, r2_before = safe_metrics(preds_test)

preds_test_corrected = pd.DataFrame({'age': y_test, 'BA_corrected': BA_te_corrected})
mae_te_after, rmse_te_after, r2_after = safe_metrics(preds_test_corrected, pred_col='BA_corrected')

metrics = {
    'mae_train': mae_tr, 'rmse_train': rmse_tr, 'r2_train': r2_tr,
    'mae_val': mae_va,   'rmse_val': rmse_va,   'r2_val': r2_va,
    'mae_test_before_correction': mae_te_before,
    'rmse_test_before_correction': rmse_te_before,
    'r^2_skl_test_before_correction': r2_before,
    'mae_test_after_correction': mae_te_after,
    'rmse_test_after_correction': rmse_te_after,
    'r^2_skl_test_after_correction': r2_after
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
    DT,
    data=X_train,                  # background for interventional E[f(x)]
    feature_perturbation="interventional"  # handles correlated EEG features correctly
)

shap = explainer(X_test, check_additivity=True)

# --- Save raw SHAP values ---
shap_cols = [f"shap_{c}" for c in feature_cols]
pd.DataFrame(shap.values, columns=shap_cols).to_csv(
    os.path.join(OUT_DIR, "shap_dt.csv"), index=False
)

# --- Plot 1: Beeswarm (test set) ---
plt.figure()
shap.plots.beeswarm(shap, max_display=20, show=False)
plt.title("TreeSHAP — Beeswarm (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_dt.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: Bar chart — mean |SHAP| (test set) ---
plt.figure()
shap.plots.bar(shap, max_display=20, show=False)
plt.title("TreeSHAP — Mean |SHAP| (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_dt.png"), dpi=150, bbox_inches='tight')
plt.show()


# --- Summary table ---
mean_abs_shap = np.abs(shap.values).mean(axis=0)

shap_summary = pd.DataFrame({
    'feature':       feature_cols,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap':     shap.values.mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_summary.to_csv(os.path.join(OUT_DIR, "shap_summary_dt.csv"), index=False)
print("\n[Top 10 Features by Mean |SHAP| — Test Set]")
print(shap_summary.head(10).to_string(index=False))
print(f"\nSHAP artifacts saved to: {OUT_DIR}")
