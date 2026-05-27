import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap

# =========================
# CONFIG SECTION
# =========================

# Dataset filepaths
TRAIN_CSV = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_train_features.csv"
VAL_CSV   = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_val_features.csv"
TEST_CSV  = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_test_features.csv"

# Set filepath for output directory
OUT_DIR = "/serverdata/ccshome/sid/Fainl Pipeline/final_outputs/06_linreg_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Random seed
SEED = 37
np.random.seed(SEED)

# 1) Load CSVs
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# 2) Determine feature columns (same reserved set)
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

X_train_raw = df_train[feature_cols].astype(float).values
X_train_raw = np.nan_to_num(X_train_raw, copy=True, nan=0.0)
y_train = df_train["age"].astype(float).values

X_val_raw = df_val[feature_cols].astype(float).values
X_val_raw = np.nan_to_num(X_val_raw, copy=True, nan=0.0)
y_val = df_val["age"].astype(float).values

X_test_raw = df_test[feature_cols].astype(float).values
X_test_raw = np.nan_to_num(X_test_raw, copy=True, nan=0.0)
y_test = df_test["age"].astype(float).values

# ==============================================================
# Scale features (fit only on train)
# ==============================================================

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# =========================
# MODEL
# =========================

# Scale features, then fit ordinary least squares linear regression.
# LinearRegression itself minimizes squared error.
LR = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# =========================
# CV EVALUATION (MSE)
# =========================

cv_scores = cross_val_score(
    LR,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

mean_cv_mse = -np.mean(cv_scores)
print(f"5-fold CV MSE: {mean_cv_mse:.6f}")

# Fit final model on full training set
LR.fit(X_train, y_train)

# Save trained model
model_path = os.path.join(OUT_DIR, "lr_model.joblib")
joblib.dump(LR, model_path)


# =========================
# PLOTTING LEARNING CURVE
# =========================

train_sizes, train_scores, val_scores = learning_curve(
    estimator=LR,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    shuffle=True,
    random_state=SEED
)

train_mse = -np.mean(train_scores, axis=1)
val_mse = -np.mean(val_scores, axis=1)

loss_plot = os.path.join(OUT_DIR, "loss.png")

plt.figure()
plt.plot(train_sizes, train_mse, label='Training MSE')
plt.plot(train_sizes, val_mse, label='Validation MSE')
plt.xlabel('Number of training samples')
plt.ylabel('MSE')
plt.title('Learning Curve (LR)')
plt.legend()
plt.savefig(loss_plot)
plt.show()


# =========================
# PREDICTIONS ON TRAIN, VAL, TEST
# =========================

BA_tr = LR.predict(X_train)
BA_va = LR.predict(X_val)
BA_te = LR.predict(X_test)

preds_train = pd.DataFrame({'age': y_train, 'BA': BA_tr, 'BAI': BA_tr - y_train})
preds_val   = pd.DataFrame({'age': y_val,   'BA': BA_va, 'BAI': BA_va - y_val})
preds_test  = pd.DataFrame({'age': y_test,   'BA': BA_te, 'BAI': BA_te - y_test})

preds_train.to_csv(os.path.join(OUT_DIR, "preds_train.csv"), index=False)
preds_val.to_csv(os.path.join(OUT_DIR, "preds_val.csv"), index=False)
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)


# =========================
# BIAS CORRECTION ON TEST SET
# =========================

# Same 10-year sliding approach
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


# =========================
# PERFORMANCE METRICS
# =========================

def safe_metrics(df, pred_col='BA', true_col='age'):
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2, mse

mae_tr, rmse_tr, r2_tr, mse_tr = safe_metrics(preds_train)
mae_va, rmse_va, r2_va, mse_va = safe_metrics(preds_val)
mae_te_before, rmse_te_before, r2_before, mse_te_before = safe_metrics(preds_test)

preds_test_corrected_for_metrics = pd.DataFrame(
    {'age': y_test, 'BA_corrected': BA_te_corrected}
)
mae_te_after, rmse_te_after, r2_after, mse_te_after = safe_metrics(
    preds_test_corrected_for_metrics, pred_col='BA_corrected'
)

metrics = {
    'mae_train': mae_tr,
    'rmse_train': rmse_tr,
    'r2_train': r2_tr,
    'mse_train': mse_tr,

    'mae_val': mae_va,
    'rmse_val': rmse_va,
    'r2_val': r2_va,
    'mse_val': mse_va,

    'mae_test_before_correction': mae_te_before,
    'rmse_test_before_correction': rmse_te_before,
    'r^2_skl_test_before_correction': r2_before,
    'mse_test_before_correction': mse_te_before,

    'mae_test_after_correction': mae_te_after,
    'rmse_test_after_correction': rmse_te_after,
    'r^2_skl_test_after_correction': r2_after,
    'mse_test_after_correction': mse_te_after
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

# ============
# LINEAR SHAP
# ============
explainer = shap.LinearExplainer(model = LR.named_steps["model"],
                                 masker = X_train,
                                 feature_perturbation = "interventional"
                                 )
# Save SHAP values as CSV
shap = explainer(X_test)
shap_cols = [f"shap_{c}" for c in feature_cols]

pd.DataFrame(shap.values, columns = shap_cols).to_csv(os.path.join(OUT_DIR,"shap_linreg.csv"), index = False)

# --- Plot 1: Beeswarm (test set) ---
# Shows feature importance + direction of effect on age prediction
plt.figure()
shap.plots.beeswarm(shap, max_display=20, show=False)
plt.title("LinearSHAP — Beeswarm (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_linreg.png"), dpi=150, bbox_inches='tight')

# --- Plot 2: Bar chart — mean |SHAP| per feature (test set) ---
plt.figure()
shap.plots.bar(shap, max_display=20, show=False)
plt.title("LinearSHAP — Mean |SHAP| (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_linreg.png"), dpi=150, bbox_inches='tight')

# --- Summary table: mean |SHAP| ranked, saved to CSV ---
mean_abs_shap = np.abs(shap.values).mean(axis=0)
shap_summary = pd.DataFrame({
    'feature':        feature_cols,
    'mean_abs_shap':  mean_abs_shap,
    'mean_shap':      shap.values.mean(axis=0),   # signed: direction of effect
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_summary.to_csv(os.path.join(OUT_DIR, "shap_summary_linreg.csv"), index=False)
print("\n[Top 10 Features by Mean |SHAP| — Test Set]")
print(shap_summary.head(10).to_string(index=False))

print(f"\nSHAP artifacts saved to: {OUT_DIR}")