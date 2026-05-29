import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn_rvm import EMRVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
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

OUT_DIR = # Output directory path for 06_rvr_outputs
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 37
np.random.seed(SEED)

N_TRIALS = 200

# =========================
# MAIN PIPELINE
# =========================

# 1) Load CSVs
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# 2) Determine feature columns
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

X_train_raw = df_train[feature_cols].astype(float).values
X_train_raw = np.nan_to_num(X_train_raw, copy=True, nan=0.0)
y_train = df_train['age'].astype(float).values

X_val_raw = df_val[feature_cols].astype(float).values
X_val_raw = np.nan_to_num(X_val_raw, copy=True, nan=0.0)
y_val = df_val['age'].astype(float).values

X_test_raw = df_test[feature_cols].astype(float).values
X_test_raw = np.nan_to_num(X_test_raw, copy=True, nan=0.0)
y_test = df_test['age'].astype(float).values

# ==============================================================
# Scale features (fit only on train)
# ==============================================================

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# =========================
# OPTUNA SEARCH
# =========================

def objective(trial):
    gamma = trial.suggest_float("gamma", 1e-5, 1e0, log=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rvr", EMRVR(
            kernel="rbf",
            gamma=gamma
        ))
    ])

    try:
        cv_scores = cross_val_score(
            model,
            X_train_raw,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        mean_mae = -np.mean(cv_scores)
    except Exception:
        return float("inf")

    return mean_mae  # Optuna minimizes MAE


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best RVR params:", study.best_params)
print("Best CV MAE:", study.best_value)

# =========================
# FINAL FIT
# =========================

best_gamma = study.best_params["gamma"]

RVR = Pipeline([
    ("scaler", StandardScaler()),
    ("rvr", EMRVR(
        kernel="rbf",
        gamma=best_gamma
    ))
])

RVR.fit(X_train_raw, y_train)

# Save best hyperparams
pd.DataFrame([{
    "gamma": best_gamma,
    "best_score": float(study.best_value)
}]).to_csv(os.path.join(OUT_DIR, "rvr_best_params.csv"), index=False)

# Save fitted scaler
fitted_scaler = RVR.named_steps["scaler"]
scaler_meta = {
    "mean": fitted_scaler.mean_,
    "scale": fitted_scaler.scale_,
    "feature_names": feature_cols
}
with open(os.path.join(OUT_DIR, "feature_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_meta, f)

# Save trained model
joblib.dump(RVR, os.path.join(OUT_DIR, "rvr_model.joblib"))

# =========================
# LEARNING CURVE
# =========================

train_sizes, train_scores, val_scores = learning_curve(
    estimator=RVR,
    X=X_train_raw,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    shuffle=True,
    random_state=SEED
)

train_mae = -np.mean(train_scores, axis=1)
val_mae = -np.mean(val_scores, axis=1)

loss_plot = os.path.join(OUT_DIR, "loss.png")
plt.figure()
plt.plot(train_sizes, train_mae, label="Training MAE")
plt.plot(train_sizes, val_mae, label="Validation MAE")
plt.xlabel("Number of training samples")
plt.ylabel("MAE")
plt.title("Learning Curve (RVR)")
plt.legend()
plt.savefig(loss_plot)
plt.show()

# =========================
# PREDICTIONS
# =========================

BA_tr = RVR.predict(X_train_raw)
BA_va = RVR.predict(X_val_raw)
BA_te = RVR.predict(X_test_raw)

preds_train = pd.DataFrame({"age": y_train, "BA": BA_tr, "BAI": BA_tr - y_train})
preds_val   = pd.DataFrame({"age": y_val,   "BA": BA_va, "BAI": BA_va - y_val})
preds_test  = pd.DataFrame({"age": y_test,   "BA": BA_te, "BAI": BA_te - y_test})

preds_train.to_csv(os.path.join(OUT_DIR, "preds_train.csv"), index=False)
preds_val.to_csv(os.path.join(OUT_DIR, "preds_val.csv"), index=False)
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)

# =========================
# BIAS CORRECTION
# =========================

starts = np.arange(18, 90, 10)
rows = []
for s in starts:
    mask = (y_train >= s) & (y_train < s + 10)
    if np.sum(mask) == 0:
        bias_train = np.nan
    else:
        bias_train = np.mean(y_train[mask] - BA_tr[mask])
    rows.append({"CA_min": int(s), "CA_max": int(s + 10), "bias": bias_train})
bias_df = pd.DataFrame(rows)

bias_for_samples = np.full_like(y_test, fill_value=np.nan, dtype=float)
for _, row in bias_df.iterrows():
    mask = (y_test >= row["CA_min"]) & (y_test <= row["CA_max"])
    bias_for_samples[mask] = row["bias"]
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
    os.path.join(OUT_DIR, "preds_test_after_bias.csv"),
    index=False
)

# =========================
# METRICS
# =========================

def safe_metrics(df, pred_col="BA", true_col="age"):
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2

mae_tr, rmse_tr, r2_tr = safe_metrics(preds_train)
mae_va, rmse_va, r2_va = safe_metrics(preds_val)
mae_te_before, rmse_te_before, r2_before = safe_metrics(preds_test)

preds_test_corrected_eval = pd.DataFrame({"age": y_test, "BA_corrected": BA_te_corrected})
mae_te_after, rmse_te_after, r2_after = safe_metrics(
    preds_test_corrected_eval,
    pred_col="BA_corrected"
)

metrics = {
    "mae_train": mae_tr, "rmse_train": rmse_tr, "r2_train": r2_tr,
    "mae_val": mae_va, "rmse_val": rmse_va, "r2_val": r2_va,
    "mae_test_before_correction": mae_te_before,
    "rmse_test_before_correction": rmse_te_before,
    "r^2_test_before_correction": r2_before,
    "mae_test_after_correction": mae_te_after,
    "rmse_test_after_correction": rmse_te_after,
    "r^2_test_after_correction": r2_after
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

preds_test["BA_corrected"] = BA_te_corrected
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test_with_correction.csv"), index=False)

with open(os.path.join(OUT_DIR, "model_metadata.pkl"), "wb") as f:
    pickle.dump(
        {
            "feature_names": feature_cols,
            "scaler_mean": fitted_scaler.mean_,
            "scaler_scale": fitted_scaler.scale_,
            "seed": SEED,
            "best_params": study.best_params,
        },
        f
    )

print("\n[Performance Summary]")
print(metrics_df.T.to_string(header=False))
print(f"\nSaved artifacts to: {OUT_DIR}")

# =========================
# KERNEL SHAP
# =========================
background = shap.kmeans(X_train,100)

rvr_step = RVR.named_steps["rvr"]
explainer = shap.KernelExplainer(rvr_step.predict, background)

shap_values = explainer.shap_values(X_test, nsamples=512)

shap_test = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_test), explainer.expected_value),
    data=X_test,
    feature_names=feature_cols
)

# --- Save raw SHAP values ---
shap_cols = [f"shap_{c}" for c in feature_cols]
pd.DataFrame(shap_values, columns=shap_cols).to_csv(
    os.path.join(OUT_DIR, "shap_rvr.csv"), index=False
)

# --- Plot 1: Beeswarm (test set) ---
plt.figure()
shap.plots.beeswarm(shap_test, max_display=20, show=False)
plt.title("KernelSHAP — Beeswarm (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_rvr.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: Bar chart — mean |SHAP| ---
plt.figure()
shap.plots.bar(shap_test, max_display=20, show=False)
plt.title("KernelSHAP — Mean |SHAP| (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_rvr.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Summary table ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary  = pd.DataFrame({
    "feature":       feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap":     shap_values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_summary.to_csv(os.path.join(OUT_DIR, "shap_summary_rvr.csv"), index=False)
print("\n[Top 10 Features by Mean |SHAP| — Test Set]")
print(shap_summary.head(10).to_string(index=False))
print(f"\nSHAP artifacts saved to: {OUT_DIR}")

