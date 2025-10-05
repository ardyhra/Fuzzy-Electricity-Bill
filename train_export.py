# train_export.py  (dengan deteksi & perbaikan nilai negatif)
import numpy as np, pandas as pd, pickle, re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CSV_PATH   = 'https://drive.google.com/uc?id=1eM01sC7_ZBHtUqk8wL0y42c1w4L7VvC7'   # ganti bila perlu
TARGET_COL = "amount_paid"

# ===== Fuzzy TSK Core =====
def fcm(X, c=3, m=2.0, max_iter=200, tol=1e-5, random_state=42):
    rng = np.random.default_rng(random_state)
    N, d = X.shape
    U = rng.random((N, c)); U = U / U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)
        dist = np.zeros((N, c))
        for k in range(c):
            diff = X - V[k]
            dist[:, k] = np.sum(diff * diff, axis=1) + 1e-12
        power = 2.0 / (m - 1.0)
        denom = (dist[:, :, None] / dist[:, None, :]) ** power
        U_new = 1.0 / np.sum(denom, axis=2)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new; break
        U = U_new
    return U, V

def compute_gaussian_params(X, U, V, m=2.0, eps=1e-6):
    N, d = X.shape; c = V.shape[0]; Um = U ** m
    sig2 = np.zeros((c, d))
    for k in range(c):
        w = Um[:, k][:, None]
        num = np.sum(w * (X - V[k])**2, axis=0)
        den = np.sum(w) + eps
        sig2[k] = num / den + eps
    return sig2

def firing_strengths(X, V, sig2):
    N, d = X.shape; c = V.shape[0]
    w = np.ones((N, c))
    for k in range(c):
        exp_term = -0.5 * ((X - V[k])**2) / (sig2[k])
        w[:, k] = np.exp(np.sum(exp_term, axis=1))
    w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12
    return w / w_sum

def build_design_matrix(X, w_norm):
    N, d = X.shape; c = w_norm.shape[1]
    Phi = np.zeros((N, c*(d+1)))
    for k in range(c):
        cols = [w_norm[:, [k]]]
        for j in range(d):
            cols.append(w_norm[:, [k]] * X[:, [j]])
        Phi[:, k*(d+1):(k+1)*(d+1)] = np.hstack(cols)
    return Phi

def fit_consequents_ls(Phi, y, ridge=1e-8):
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    b = Phi.T @ y
    return np.linalg.solve(A, b)

def predict_tsk(X, V, sig2, theta):
    w_norm = firing_strengths(X, V, sig2)
    Phi = build_design_matrix(X, w_norm)
    return Phi @ theta

# ===== Load & Constraints =====
df = pd.read_csv(CSV_PATH)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != TARGET_COL]

# RULE: heuristik constraints & units
def infer_unit(col: str):
    c = col.lower()
    if "area" in c: return "m²"
    if "room" in c or "kamar" in c: return "unit"
    if "people" in c or "person" in c or "penghuni" in c: return "orang"
    if "children" in c or "anak" in c: return "orang"
    if "income" in c or "pendapatan" in c: return "Rp/bulan"
    if "ac" in c or "tv" in c or "flat" in c or "urban" in c: return "(0/1)"
    return ""

def infer_constraint(col: str):
    c = col.lower()
    # integer-like count
    if any(k in c for k in ["room", "kamar", "people", "person", "children", "anak"]):
        return dict(min_allowed=0.0, enforce_int=True)
    # binary flags
    if any(k in c for k in ["is_", "ac", "tv", "flat", "urban"]):
        return dict(min_allowed=0.0, enforce_int=True, binary=True)
    # money / area or other positive numeric
    if any(k in c for k in ["income", "pendapatan", "area"]):
        return dict(min_allowed=0.0, enforce_int=False)
    # default: non-negative
    return dict(min_allowed=0.0, enforce_int=False)

UNITS = {col: infer_unit(col) for col in feature_cols}
CONSTRAINTS = {col: infer_constraint(col) for col in feature_cols}

# imputasi median sederhana
data = df[feature_cols + [TARGET_COL]].copy()
for c in feature_cols + [TARGET_COL]:
    if data[c].isna().any():
        data[c] = data[c].fillna(data[c].median())

# DETEKSI & PERBAIKAN NEGATIF (clamp ke min_allowed)
fixed_counts = {}
for col in feature_cols:
    rule = CONSTRAINTS[col]
    min_allowed = rule.get("min_allowed", 0.0)
    before = (data[col] < min_allowed).sum()
    if before > 0:
        data.loc[data[col] < min_allowed, col] = min_allowed
    fixed_counts[col] = int(before)
    # opsional enforce int utk count/binary
    if rule.get("enforce_int", False):
        data[col] = data[col].round().astype(int)

# Info ringkas perbaikan
total_fixed = sum(fixed_counts.values())
if total_fixed > 0:
    print("Perbaikan nilai < min_allowed (clamp) dilakukan:")
    for col, cnt in fixed_counts.items():
        if cnt > 0:
            print(f"  - {col}: {cnt} nilai dikoreksi ke {CONSTRAINTS[col]['min_allowed']}")

# train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_test  = scaler.transform(test_df[feature_cols].values)
y_train = train_df[TARGET_COL].values.astype(float)
y_test  = test_df[TARGET_COL].values.astype(float)

# pilih c terbaik sederhana
best = {"rmse": np.inf}
for c in [2,3,4,5]:
    U, V = fcm(X_train, c=c, m=2.0, random_state=42)
    sig2 = compute_gaussian_params(X_train, U, V, m=2.0)
    w_norm = firing_strengths(X_train, V, sig2)
    Phi = build_design_matrix(X_train, w_norm)
    theta = fit_consequents_ls(Phi, y_train)
    y_pred = predict_tsk(X_test, V, sig2, theta)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    if rmse < best["rmse"]:
        best = dict(c=c, V=V, sig2=sig2, theta=theta, rmse=rmse)

print(f"Best c={best['c']} | RMSE={best['rmse']:.4f}")

# ranges real dari TRAIN untuk validasi GUI
ranges = {}
for col in feature_cols:
    v = train_df[col].astype(float)
    ranges[col] = {"min": float(np.nanmin(v)), "max": float(np.nanmax(v))}

artifact = {
    "feature_cols": feature_cols,
    "target_col": TARGET_COL,
    "scaler": scaler,
    "model": {"V": best["V"], "sig2": best["sig2"], "theta": best["theta"]},
    "meta_types": {col: ("binary" if CONSTRAINTS[col].get("binary") else ("numeric_int" if CONSTRAINTS[col].get("enforce_int") else "numeric_float")) for col in feature_cols},
    "ranges": ranges,      # rentang real dari TRAIN
    "units": UNITS,        # label satuan
    "constraints": CONSTRAINTS,           # <-- BARU: aturan min_allowed & integer/binary
    "scaler_range": {                     # <-- BARU: untuk clamp sebelum scaling
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
    }
}

with open("fuzzy_tsk_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("Saved fuzzy_tsk_model.pkl")
