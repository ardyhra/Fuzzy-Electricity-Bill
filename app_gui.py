# app_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np, pickle, os, sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====== TSK Core ======
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

def predict_tsk_full(X, V, sig2, theta):
    w_norm = firing_strengths(X, V, sig2)
    Phi = build_design_matrix(X, w_norm)
    y_hat = (Phi @ theta).ravel()
    N, d = X.shape; c = V.shape[0]
    y_locals = np.zeros((N, c))
    for k in range(c):
        start = k*(d+1); end = (k+1)*(d+1)
        params = theta[start:end]
        y_locals[:, k] = params[0] + X @ params[1:]
    return y_hat, w_norm, y_locals

# ====== Resource path (PyInstaller) ======
def resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

# ====== Load artifact ======
def load_artifact(pkl_path):
    with open(pkl_path, "rb") as f:
        art = pickle.load(f)
    for key in ["feature_cols","scaler","model","meta_types","ranges","units","constraints","scaler_range"]:
        if key not in art:
            raise ValueError(f"Artifact missing key: {key}")
    return art

DEFAULT_MODEL = resource_path("fuzzy_tsk_model.pkl")
artifact = None
try:
    artifact = load_artifact(DEFAULT_MODEL)
except Exception as e:
    print("Note:", e)

# ====== GUI ======
root = tk.Tk()
root.title("Prediksi Tagihan Listrik - Fuzzy TSK")
root.geometry("900x720")

# Menu
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)

def on_load_model():
    global artifact, FEATURES, SCALER, V, SIG2, THETA, META, RANGES, UNITS, CONSTRAINTS, SR_MIN, SR_MAX
    p = filedialog.askopenfilename(title="Pilih model .pkl",
        filetypes=[("Pickle files","*.pkl"),("All files","*.*")])
    if not p: return
    try:
        artifact = load_artifact(p)
        FEATURES = artifact["feature_cols"]
        SCALER   = artifact["scaler"]
        V        = artifact["model"]["V"]
        SIG2     = artifact["model"]["sig2"]
        THETA    = artifact["model"]["theta"]
        META     = artifact["meta_types"]
        RANGES   = artifact["ranges"]
        UNITS    = artifact["units"]
        CONSTRAINTS = artifact["constraints"]
        SR_MIN   = np.array(artifact["scaler_range"]["data_min_"], dtype=float)
        SR_MAX   = np.array(artifact["scaler_range"]["data_max_"], dtype=float)
        rebuild_inputs()
        messagebox.showinfo("Model Loaded", f"Berhasil memuat:\n{os.path.basename(p)}")
    except Exception as e:
        messagebox.showerror("Load Model Error", str(e))

filemenu.add_command(label="Load Model…", command=on_load_model)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

# Header
header = ttk.Frame(root, padding=10)
header.pack(fill="x")
ttk.Label(header, text="Prediksi Tagihan Listrik (amount_paid) — Fuzzy TSK",
          font=("Segoe UI", 14, "bold")).pack(side="left")

# Panel utama
main = ttk.Frame(root, padding=10); main.pack(fill="both", expand=True)
left = ttk.Frame(main); left.pack(side="left", fill="y", padx=(0,10))
right = ttk.Frame(main); right.pack(side="left", fill="both", expand=True)

# Input panel
inputs_frame = ttk.LabelFrame(left, text="Input Fitur"); inputs_frame.pack(fill="x", pady=5)
entries = {}; units_labels = {}; range_labels = {}

# Auto-clamp toggle
opts = ttk.LabelFrame(left, text="Opsi Validasi"); opts.pack(fill="x", pady=6)
auto_clamp_var = tk.BooleanVar(value=True)
ttk.Checkbutton(opts, text="Auto-Clamp ke rentang valid (min_allowed, data train, & scaler)",
                variable=auto_clamp_var).pack(anchor="w", padx=6, pady=4)

# Hasil
result_frame = ttk.LabelFrame(left, text="Hasil Prediksi"); result_frame.pack(fill="x", pady=10)
pred_var = tk.StringVar(value="—")
ttk.Label(result_frame, text="Perkiraan amount_paid:", width=24).pack(side="left", padx=(6,2), pady=6)
ttk.Label(result_frame, textvariable=pred_var, font=("Segoe UI", 12, "bold")).pack(side="left", pady=6)

# Info/status
status = ttk.Label(left, text="", foreground="#777"); status.pack(fill="x", pady=(4,2))

def rebuild_inputs():
    for w in inputs_frame.winfo_children():
        w.destroy()
    entries.clear(); units_labels.clear(); range_labels.clear()

    for col in FEATURES:
        row = ttk.Frame(inputs_frame); row.pack(fill="x", pady=4)
        ttk.Label(row, text=col, width=24).pack(side="left")
        meta = META.get(col, "numeric_float")
        u = UNITS.get(col, "")
        rng = RANGES.get(col, None)

        if meta == "binary":
            var = tk.IntVar(value=0)
            ttk.Checkbutton(row, variable=var, text="Ya (0/1)").pack(side="left")
            entries[col] = var
            ttk.Label(row, text=u or "(0/1)", foreground="#666").pack(side="left", padx=6)
            if rng:
                ttk.Label(row, text=f"[{int(rng['min'])}-{int(rng['max'])}]", foreground="#888").pack(side="left", padx=6)
        else:
            var = tk.StringVar(value="")
            ttk.Entry(row, textvariable=var, width=14).pack(side="left")
            entries[col] = var
            ttk.Label(row, text=u, foreground="#666").pack(side="left", padx=6)
            if rng:
                ttk.Label(row, text=f"[{rng['min']:.2f} .. {rng['max']:.2f}]", foreground="#888").pack(side="left", padx=6)

# Tombol
btns = ttk.Frame(left); btns.pack(fill="x", pady=6)

def on_predict():
    if artifact is None:
        messagebox.showwarning("Model", "Model belum dimuat. Gunakan File → Load Model…")
        return

    clamps = []  # catat clamp yang terjadi
    raw = []
    for i, col in enumerate(FEATURES):
        meta = META.get(col, "numeric_float")
        cons = CONSTRAINTS.get(col, {})
        min_allowed = cons.get("min_allowed", 0.0)

        if meta == "binary":
            val = int(entries[col].get())
            if auto_clamp_var.get():
                if val < min_allowed: 
                    clamps.append(f"{col}: {val}→{int(min_allowed)} (min_allowed)")
                    val = int(min_allowed)
                # juga clamp ke [0,1] jika binary
                if val < 0: 
                    clamps.append(f"{col}: {val}→0 (binary)")
                    val = 0
                if val > 1:
                    clamps.append(f"{col}: {val}→1 (binary)")
                    val = 1
            else:
                if val < min_allowed:
                    messagebox.showerror("Validasi", f"'{col}' tidak boleh < {min_allowed}.")
                    return
            raw.append(val)

        else:
            txt = entries[col].get().strip()
            if txt == "":
                messagebox.showerror("Input", f"Fitur '{col}' belum diisi.")
                return
            try:
                val = float(txt)
            except:
                messagebox.showerror("Input", f"Fitur '{col}' harus berupa angka.")
                return

            if auto_clamp_var.get():
                if val < min_allowed:
                    clamps.append(f"{col}: {val}→{min_allowed} (min_allowed)")
                    val = min_allowed
                rng = RANGES.get(col, None)
                if rng:
                    if val < rng["min"]:
                        clamps.append(f"{col}: {val}→{rng['min']} (train_min)")
                        val = rng["min"]
                    if val > rng["max"]:
                        clamps.append(f"{col}: {val}→{rng['max']} (train_max)")
                        val = rng["max"]
            else:
                if val < min_allowed:
                    messagebox.showerror("Validasi", f"'{col}' tidak boleh < {min_allowed}.")
                    return
            raw.append(val)

    X_real = np.array([raw], dtype=float)

    # CLAMP SEBELUM SCALER (agar tidak negatif/ >1 setelah MinMax)
    sr_min = SR_MIN
    sr_max = SR_MAX
    if auto_clamp_var.get():
        X_clamped = np.minimum(np.maximum(X_real, sr_min), sr_max)
        # catat jika ada clamp scaler
        diff = (X_clamped != X_real)
        if diff.any():
            for j, col in enumerate(FEATURES):
                if diff[0, j]:
                    clamps.append(f"{col}: clamp to scaler_range [{sr_min[j]:.4g}..{sr_max[j]:.4g}]")
        X_use = X_clamped
    else:
        X_use = X_real

    X_scaled = SCALER.transform(X_use)
    y_hat, w_norm, y_locals = predict_tsk_full(X_scaled, V, SIG2, THETA)

    pred_var.set(f"{float(y_hat[0]):,.2f}")

    # update grafis
    plot_rule_weights(w_norm[0])
    plot_rule_locals(y_locals[0])

    # tampilkan info clamp (kalau ada)
    if clamps:
        status.config(text="Auto-Clamp: " + "; ".join(clamps))
        messagebox.showwarning("Auto-Clamp", "Beberapa nilai dikoreksi ke rentang valid:\n- " + "\n- ".join(clamps))
    else:
        status.config(text="OK")

def on_reset():
    for col in FEATURES:
        if META.get(col) == "binary":
            entries[col].set(0)
        else:
            entries[col].set("")
    pred_var.set("—")
    status.config(text="")
    plot_rule_weights(None, clear=True)
    plot_rule_locals(None, clear=True)

ttk.Button(btns, text="Prediksi", command=on_predict).pack(side="left", padx=4)
ttk.Button(btns, text="Reset", command=on_reset).pack(side="left", padx=4)

# Plot area
plots = ttk.LabelFrame(right, text="Grafik Kontribusi Aturan"); plots.pack(fill="both", expand=True)
fig = Figure(figsize=(7.0, 5.2), dpi=100)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
fig.tight_layout(pad=2.0)
canvas = FigureCanvasTkAgg(fig, master=plots)
canvas.get_tk_widget().pack(fill="both", expand=True)

def plot_rule_weights(w, clear=False):
    ax1.clear()
    ax1.set_title("Kontribusi Aturan (w̄)"); ax1.set_ylabel("w̄"); ax1.set_ylim(0, 1)
    if not clear and w is not None:
        idx = np.arange(len(w))
        ax1.bar([f"Rule-{i+1}" for i in idx], w)
    canvas.draw()

def plot_rule_locals(y, clear=False):
    ax2.clear()
    ax2.set_title("Keluaran Lokal y_k(x)"); ax2.set_ylabel("y_k(x)")
    if not clear and y is not None:
        idx = np.arange(len(y))
        ax2.bar([f"Rule-{i+1}" for i in idx], y)
    canvas.draw()

# Bind artifact default
if artifact:
    FEATURES = artifact["feature_cols"]
    SCALER   = artifact["scaler"]
    V        = artifact["model"]["V"]
    SIG2     = artifact["model"]["sig2"]
    THETA    = artifact["model"]["theta"]
    META     = artifact["meta_types"]
    RANGES   = artifact["ranges"]
    UNITS    = artifact["units"]
    CONSTRAINTS = artifact["constraints"]
    SR_MIN   = np.array(artifact["scaler_range"]["data_min_"], dtype=float)
    SR_MAX   = np.array(artifact["scaler_range"]["data_max_"], dtype=float)
else:
    FEATURES = []; SCALER = V = SIG2 = THETA = META = RANGES = UNITS = CONSTRAINTS = None
    SR_MIN = SR_MAX = None

rebuild_inputs()
plot_rule_weights(None, clear=True)
plot_rule_locals(None, clear=True)

root.mainloop()
