import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import gpytorch
from models.model_gpr_gpy import GPYModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Dataset filepaths
TRAIN_CSV = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_train_features.csv"
VAL_CSV   = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_val_features.csv"
TEST_CSV  = "/serverdata/ccshome/sid/Catch22/Catch22_Features/C22_test_features.csv"

# Set filepath for output directory
OUT_DIR = "/serverdata/ccshome/sid/Final Pipeline/final_outputs/06_gpy_gpr_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Seed for reproducibility
SEED = 37
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training hyperparameters
EPOCHS = 350
LR = 1e-2

# Device
device = "cpu"

df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

# Determine feature columns
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

# Extract arrays
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
# 2. Scale features (fit only on train)
# ==============================================================

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# Save scaler for future use
scaler_path = os.path.join(OUT_DIR, "feature_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(
        {"mean": scaler.mean_,
         "scale": scaler.scale_,
         "feature_names": feature_cols},
        f
    )

# Convert to torch tensors (full batch, no DataLoader)
Xtr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
ytr_t = torch.tensor(y_train, dtype=torch.float32, device=device)

Xva_t = torch.tensor(X_val, dtype=torch.float32, device=device)
yva_t = torch.tensor(y_val, dtype=torch.float32, device=device)

Xte_t = torch.tensor(X_test, dtype=torch.float32, device=device)
yte_t = torch.tensor(y_test, dtype=torch.float32, device=device)

ard_num_dims = Xtr_t.shape[1]

# ==============================================================
# 3. Instantiate GPY GP model
# ==============================================================

model, likelihood = GPYModel.build_model_and_likelihood(Xtr_t, ytr_t, kernel_name = "RBF", ard_num_dims=ard_num_dims)
model = model.to(device).to(torch.float32)
likelihood = likelihood.to(device).to(torch.float32)
print("Imported build_model_and_likelihood from model.py")
# model.set_train_data(inputs=(Xtr_t,), targets=ytr_t, strict=False)

# 4) Optimiser and Marginal Log Likelihood (MLL)
model.train()
likelihood.train()
optimiser = torch.optim.Adam(model.parameters(), lr = LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# 5) Train Loop

nll_history = [] # Negative Log Likelihood (NLL)
val_mse_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    likelihood.train()
    optimiser.zero_grad()
    with gpytorch.settings.debug(False):   # disables the train-input check
        output = model(Xtr_t)
    loss = -mll(output,ytr_t) # minimising negative marginal log likelihood
    loss.backward()
    optimiser.step()

    nll_val = float(loss.detach().cpu().numpy())
    nll_history.append(nll_val)

    # Validation for monitoring
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        BA_val_mean = likelihood(model(Xva_t)).mean
        mse_val = torch.mean((BA_val_mean - yva_t) ** 2).item()
        val_mse_history.append(mse_val)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"NLML: {nll_val:.6f} | "
        f"Val MSE: {mse_val:.6f}"
    )
# ==============================================================
# 5. Plot loss curves (NLML + val MSE)
# ==============================================================

loss_plot = os.path.join(OUT_DIR, "loss.png")
plt.figure(figsize=(7, 4))
plt.plot(nll_history, label="negative log marginal likelihood")
plt.plot(val_mse_history, label="val MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(loss_plot)
plt.show()
plt.close()

# ==============================================================
# 6. Save trained GP model
# ==============================================================

model_path = os.path.join(OUT_DIR, "gpy_gpr_model.pt")
# Uses the custom save() method defined in ExactGPR; this includes hyperparams
torch.save({
    "model_state_dict": model.state_dict(),
    "likelihood_state_dict": likelihood.state_dict()
}, model_path)
print(f"Saved trained Exact GPR model to {model_path}")

# ==============================================================
# 7. Evaluate on train/val/test sets
# ==============================================================

model.eval()
with torch.no_grad():
    BA_tr_mean, BA_tr_var = model.predict(Xtr_t, return_var=True)
    BA_va_mean, BA_va_var = model.predict(Xva_t, return_var=True)
    BA_te_mean, BA_te_var = model.predict(Xte_t, return_var=True)

BA_tr = BA_tr_mean.cpu().numpy()
BA_va = BA_va_mean.cpu().numpy()
BA_te = BA_te_mean.cpu().numpy()

# Prediction DataFrames
preds_train = pd.DataFrame(
   {"age": y_train, "BA": BA_tr, "BAI": BA_tr - y_train}
)
preds_val = pd.DataFrame(
    {"age": y_val, "BA": BA_va, "BAI": BA_va - y_val}
)
preds_test = pd.DataFrame(
    {"age": y_test, "BA": BA_te, "BAI": BA_te - y_test}
)

# preds_train.to_csv(os.path.join(OUT_DIR, "preds_train.csv"), index=False)
preds_val.to_csv(os.path.join(OUT_DIR, "preds_val.csv"), index=False)
preds_test.to_csv(os.path.join(OUT_DIR, "preds_test_before_bias.csv"), index=False)

# ==============================================================
# 8. Bias correction (same 5-year sliding window logic as before)
# ==============================================================
# Bias Correction
# Compute 10-year sliding bias table for train set
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

# ==============================================================
# 9. Metrics
# ==============================================================

def safe_metrics(df, pred_col='BA', true_col='age'):
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

mae_te_after, rmse_te_after, r2_after = safe_metrics(
    pd.DataFrame({"age": y_test, "BA_corrected": BA_te_corrected}),
    pred_col="BA_corrected",
)

metrics = {
    "mae_train": mae_tr,
    "rmse_train": rmse_tr,
    "r2_train": r2_tr,
    "mae_val": mae_va,
    "rmse_val": rmse_va,
    "r2_val": r2_va,
    "mae_test_before_correction": mae_te_before,
    "rmse_test_before_correction": rmse_te_before,
    "r^2_test_before_correction": r2_before,
    "mae_test_after_correction": mae_te_after,
    "rmse_test_after_correction": rmse_te_after,
    "r^2_test_after_correction": r2_after,
    "epochs": EPOCHS,
    "lr": LR,
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(OUT_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\n[Performance Summary]")
print(metrics_df.T.to_string(header=False))


# =========================
# KERNEL SHAP
# =========================
background = shap.kmeans(X_train,100)

explainer = shap.KernelExplainer(model,background)

shap_values = explainer.shap_values(X_test, nsamples = 512)

shap = shap.Explanation(values = shap_values,
                        base_values = np.full(len(X_test), explainer.expected_value),
                        data = X_test,
                        feature_names = feature_cols
)


# --- Save raw SHAP values ---
shap_cols = [f"shap_{c}" for c in feature_cols]
pd.DataFrame(shap_values, columns=shap_cols).to_csv(
    os.path.join(OUT_DIR, "shap_gpr.csv"), index=False
)

# --- Plot 1: Beeswarm (test set) ---
plt.figure()
shap.plots.beeswarm(shap, max_display=20, show=False)
plt.title("KernelSHAP — Beeswarm (Test Set, GPR)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_gpr.png"), dpi=150, bbox_inches='tight')

# --- Plot 2: Bar chart — mean |SHAP| ---
plt.figure()
shap.plots.bar(shap, max_display=20, show=False)
plt.title("KernelSHAP — Mean |SHAP| (Test Set, GPR)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_gpr.png"), dpi=150, bbox_inches='tight')

# --- Summary table ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary  = pd.DataFrame({
    "feature":       feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap":     shap_values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_summary.to_csv(os.path.join(OUT_DIR, "shap_summary_gpr.csv"), index=False)
print("\n[Top 10 Features by Mean |SHAP| — Test Set]")
print(shap_summary.head(10).to_string(index=False))
print(f"\nSHAP artifacts saved to: {OUT_DIR}")

