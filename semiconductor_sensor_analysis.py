"""
=============================================================================
PROJECT: Semiconductor Production Test Data Analysis Pipeline
Author  : Sanusi Isiaka Olatunji
Context : Infineon Technologies Data Science Internship alignment
Pipeline: Generation → Preprocessing → Anomaly Detection → Visualisation
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── 1. SYNTHETIC DATASET ────────────────────────────────────────────────────
N_SAMPLES, N_ANOMALIES = 1_200, 60
print("="*65)
print("  Semiconductor Production Test Data Analysis Pipeline")
print("="*65)
print(f"\n[1/4] Generating dataset ({N_SAMPLES} chips)...")

nominal = {
    "VDD_mV":       (1800, 30),
    "IDD_uA":       (250,  20),
    "TEMP_C":       (27,    8),
    "FREQ_MHz":     (100,   5),
    "LEAKAGE_nA":   (15,    4),
    "VOUT_mV":      (3300, 50),
    "RISE_TIME_ns": (2.5, 0.3),
}
FEATURES = list(nominal.keys())

df = pd.DataFrame({c: np.random.normal(m, s, N_SAMPLES) for c,(m,s) in nominal.items()})

# Inject anomalies
ai = np.random.choice(N_SAMPLES, N_ANOMALIES, replace=False)
df.loc[ai, "VDD_mV"]       = df.loc[ai, "VDD_mV"]       + np.random.choice([-200,220], N_ANOMALIES)
df.loc[ai, "IDD_uA"]       = df.loc[ai, "IDD_uA"]       * np.random.uniform(2.5, 4.0, N_ANOMALIES)
df.loc[ai, "LEAKAGE_nA"]   = df.loc[ai, "LEAKAGE_nA"]   * np.random.uniform(5.0,10.0, N_ANOMALIES)
df.loc[ai, "RISE_TIME_ns"] = df.loc[ai, "RISE_TIME_ns"] + np.random.uniform(3, 8, N_ANOMALIES)

# Missing values (3%)
miss_mask = np.random.rand(*df.shape) < 0.03
df_raw = df.copy()
for r,c in zip(*np.where(miss_mask)):
    df_raw.iloc[r, c] = np.nan

print(f"   Shape: {df_raw.shape}, Missing: {df_raw.isna().sum().sum()}")

# ── 2. PREPROCESSING ────────────────────────────────────────────────────────
print("\n[2/4] Preprocessing...")
df_clean = df_raw[FEATURES].copy()

for col in FEATURES:
    med = df_clean[col].median()
    n_m = int(df_clean[col].isna().sum())
    df_clean[col] = df_clean[col].fillna(med)
    if n_m: print(f"   Imputed {n_m:>3d} missing in '{col}' (median={med:.2f})")

for col in FEATURES:
    lo, hi = df_clean[col].quantile([0.01, 0.99])
    df_clean[col] = df_clean[col].clip(lo, hi)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_clean.values)
assert not np.isnan(X_scaled).any()
print("   ✓ Imputation (median), Winsorisation [1%–99%], StandardScaler")

# ── 3. ANOMALY DETECTION ────────────────────────────────────────────────────
print("\n[3/4] Anomaly detection...")
iso = IsolationForest(contamination=0.05, random_state=SEED, n_estimators=200)
iso_lbl   = iso.fit_predict(X_scaled)
iso_score = iso.decision_function(X_scaled)

z_scores = np.abs(stats.zscore(X_scaled, axis=0))
z_flag   = (z_scores > 3.0).any(axis=1)

combined = (iso_lbl == -1) | z_flag
df_clean["anomaly"]       = combined
df_clean["anomaly_score"] = -iso_score

n_iso = int((iso_lbl==-1).sum())
n_z   = int(z_flag.sum())
n_all = int(combined.sum())
print(f"   Isolation Forest: {n_iso} | Z-Score: {n_z} | Union: {n_all} ({n_all/N_SAMPLES*100:.1f}%)")

# ── 4. VISUALISATION ────────────────────────────────────────────────────────
print("\n[4/4] Building visualisations...")

CB="#0d1117"; CP="#161b22"; CBR="#30363d"
CT="#e6edf3"; CM="#8b949e"; CN="#58a6ff"; CA="#f85149"; CG="#3fb950"

plt.rcParams.update({
    "figure.facecolor":CB,"axes.facecolor":CP,"axes.edgecolor":CBR,
    "axes.labelcolor":CT,"xtick.color":CM,"ytick.color":CM,
    "text.color":CT,"grid.color":CBR,"grid.linestyle":"--",
    "grid.linewidth":0.5,"font.family":"monospace"
})

fig = plt.figure(figsize=(20,22),facecolor=CB)
fig.suptitle(
    "Semiconductor Production Test — Data Analysis Report\n"
    "Sanusi Isiaka Olatunji  |  M.Sc. Data Science  |  University of Leoben",
    fontsize=14, color=CT, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38)

nm = ~df_clean["anomaly"]
am =  df_clean["anomaly"]

# (A) Distributions
for i, col in enumerate(FEATURES):
    row, c = divmod(i, 4)
    ax = fig.add_subplot(gs[row, c])
    ax.hist(df_clean.loc[nm, col], bins=35, color=CN, alpha=0.75, label="Normal",  density=True)
    ax.hist(df_clean.loc[am, col], bins=15, color=CA, alpha=0.80, label="Anomaly", density=True)
    ax.set_title(col, fontsize=9, color=CT, pad=4)
    ax.set_ylabel("Density", fontsize=7, color=CM)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, framealpha=0.2, labelcolor=CT, loc="upper right")

# (B) Correlation heatmap
ax_h = fig.add_subplot(gs[2, :2])
corr = df_clean[FEATURES].corr()
im   = ax_h.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_h.set_xticks(range(len(FEATURES))); ax_h.set_yticks(range(len(FEATURES)))
ax_h.set_xticklabels([f.replace("_","\n") for f in FEATURES], fontsize=6)
ax_h.set_yticklabels(FEATURES, fontsize=6)
ax_h.set_title("Feature Correlation Matrix", fontsize=10, color=CT, pad=8)
for r in range(len(FEATURES)):
    for c in range(len(FEATURES)):
        ax_h.text(c, r, f"{corr.values[r,c]:.2f}", ha="center", va="center",
                  fontsize=5.5, color="white" if abs(corr.values[r,c])>0.5 else CM)
plt.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)

# (C) Anomaly score distribution
ax_s = fig.add_subplot(gs[2, 2:])
ax_s.hist(df_clean.loc[nm, "anomaly_score"], bins=40, color=CN, alpha=0.8, label="Normal",  density=True)
ax_s.hist(df_clean.loc[am, "anomaly_score"], bins=20, color=CA, alpha=0.9, label="Anomaly", density=True)
ax_s.axvline(df_clean["anomaly_score"].quantile(0.95), color=CG, lw=1.5, ls="--", label="95th pct")
ax_s.set_title("Isolation Forest Anomaly Score Distribution", fontsize=10, color=CT)
ax_s.set_xlabel("Anomaly Score", fontsize=8); ax_s.set_ylabel("Density", fontsize=8)
ax_s.legend(fontsize=8, framealpha=0.2, labelcolor=CT); ax_s.grid(True, alpha=0.4)

# (D) PCA map
ax_p = fig.add_subplot(gs[3, :2])
pca   = PCA(n_components=2, random_state=SEED)
Xp    = pca.fit_transform(X_scaled)
ax_p.scatter(Xp[nm,0], Xp[nm,1], c=CN, alpha=0.35, s=12, label="Normal")
ax_p.scatter(Xp[am,0], Xp[am,1], c=CA, alpha=0.90, s=40, marker="x", linewidths=1.5, label="Anomaly")
ax_p.set_title(
    f"PCA Anomaly Map  (PC1={pca.explained_variance_ratio_[0]*100:.1f}%,"
    f" PC2={pca.explained_variance_ratio_[1]*100:.1f}%)",
    fontsize=9, color=CT)
ax_p.set_xlabel("PC1",fontsize=8); ax_p.set_ylabel("PC2",fontsize=8)
ax_p.legend(fontsize=8, framealpha=0.2, labelcolor=CT); ax_p.grid(True, alpha=0.4)

# (E) Feature sensitivity
ax_b = fig.add_subplot(gs[3, 2:])
sens = [abs(df_clean.loc[am,c].mean() - df_clean.loc[nm,c].mean()) /
        (df_clean[c].std()+1e-9) for c in FEATURES]
bars = ax_b.barh(FEATURES, sens, color=CA, alpha=0.75, height=0.6)
ax_b.set_xlabel("Mean Deviation (σ) Normal vs Anomalous", fontsize=8)
ax_b.set_title("Feature Sensitivity to Anomalies", fontsize=10, color=CT)
ax_b.tick_params(axis="y", labelsize=8); ax_b.grid(True, axis="x", alpha=0.4)
for bar, val in zip(bars, sens):
    ax_b.text(val+0.05, bar.get_y()+bar.get_height()/2, f"{val:.2f}σ",
              va="center", fontsize=7, color=CT)

out = "/mnt/user-data/outputs/semiconductor_analysis_report.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=CB)
print(f"   Saved → {out}")

print("\n"+"="*65)
print("  SUMMARY")
print("="*65)
print(f"  Chips tested            : {N_SAMPLES:,}")
print(f"  Features                : {len(FEATURES)}")
print(f"  Missing values imputed  : {int(miss_mask.sum()):,}")
print(f"  Anomalies detected      : {n_all}  ({n_all/N_SAMPLES*100:.1f}%)")
print(f"  Top sensitive feature   : {FEATURES[int(np.argmax(sens))]}")
print(f"  PCA variance (2 PCs)    : {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
print("="*65)
print("  Pipeline complete ✓")
print("="*65)
