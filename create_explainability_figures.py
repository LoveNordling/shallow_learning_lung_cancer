#!/usr/bin/env python3
"""
create_explainability_figures2.py

Behavior (like your old script, but CI-aware):
- Run with no arguments: build two grids (Weights, Permutation) for the 7 canonical
  experiment combinations and one combined "double" figure (Weights vs Permutation)
  for 'clinical parameters_pleomorphism_densities' if present.

Adds:
- If per-split files exist (…_per_split.csv), compute mean ±95% t-CI and p-values vs 0
  (Shapiro -> t-test else Wilcoxon) with BH-FDR (* on plot for q<0.05).
- If only mean files exist, plot as before (no CI/error bars).

Loud/debuggable: no broad try/except; prints discovered inputs and written outputs.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, ttest_1samp, shapiro, wilcoxon

PLOT_DIR = "./plots"

# Canonical set and order for grids
DESIRED_ORDER = [
    "clinical parameters",
    "densities",
    "pleomorphism",
    "clinical parameters_densities",
    "clinical parameters_pleomorphism",
    "pleomorphism_densities",
    "clinical parameters_pleomorphism_densities",
]

EXP_FULLNAME_MAP = {
    "clinical parameters": "Clinical Parameters",
    "densities": "Immune Cell Densities",
    "pleomorphism": "Pleomorphism",
    "clinical parameters_densities": "Clinical Parameters + Densities",
    "clinical parameters_pleomorphism": "Clinical Parameters + Pleomorphism",
    "pleomorphism_densities": "Densities + Pleomorphism",
    "clinical parameters_pleomorphism_densities": "Clinical Parameters + Densities + Pleomorphism",
}

# ---------------- Stats helpers ----------------
def clean_feature_name(name: str):
    if not isinstance(name, str):
        return name
    name = name.replace("_", " ")
    if name == "Gender":
        name = "Sex (Male)"
    return name

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = min(p[idx] * n / rank, prev, 1.0)
        adj[idx] = val
        prev = val
    return adj

def summarize_vs_zero(values):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    mean = float(np.mean(x)) if n else np.nan
    if n < 2:
        return dict(mean=mean, ci_lo=np.nan, ci_hi=np.nan, p_value=np.nan)
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    ci_lo, ci_hi = mean - tcrit * sem, mean + tcrit * sem
    # pick test
    p_norm = shapiro(x).pvalue if 3 < n < 5000 else 1.0
    if p_norm < 0.05:
        p_val = wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue
    else:
        p_val = ttest_1samp(x, 0.0, alternative="two-sided").pvalue
    return dict(mean=mean, ci_lo=ci_lo, ci_hi=ci_hi, p_value=float(p_val))

# ---------------- Data loading ----------------
def load_stats(exp, kind):
    """kind in {'weights','perm'}; returns df with mean, CI, p, has_ci."""
    suffix_split = "_feature_importance_per_split.csv" if kind == "weights" else "_perm_importance_per_split.csv"
    suffix_mean  = "_feature_importance.csv"            if kind == "weights" else "_perm_importance.csv"
    path_split = os.path.join(PLOT_DIR, f"{exp}_LogReg{suffix_split}")
    path_mean  = os.path.join(PLOT_DIR, f"{exp}_LogReg{suffix_mean}")

    print(f"[load] {exp:>40s} ({kind})")
    print(f"       per-split: {os.path.relpath(path_split)}   exists={os.path.exists(path_split)}")
    print(f"       mean-only: {os.path.relpath(path_mean)}    exists={os.path.exists(path_mean)}")

    if os.path.exists(path_split):
        df = pd.read_csv(path_split)
        rows = []
        for f, sub in df.groupby("feature"):
            stats = summarize_vs_zero(sub["importance"].values)
            rows.append({"feature": f, "mean": stats["mean"],
                         "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
                         "p_value": stats["p_value"]})
        out = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
        out["feature_pretty"] = [clean_feature_name(f) for f in out["feature"]]
        out["p_adj"] = bh_fdr(out["p_value"].fillna(1.0).values)
        out["has_ci"] = True
        return out

    if os.path.exists(path_mean):
        df = pd.read_csv(path_mean).rename(columns={"importance": "mean"})
        df["feature_pretty"] = [clean_feature_name(f) for f in df["feature"]]
        df["ci_lo"] = np.nan
        df["ci_hi"] = np.nan
        df["p_value"] = np.nan
        df["p_adj"] = np.nan
        df["has_ci"] = False
        return df

    raise FileNotFoundError(f"No explainability files for {exp} ({kind}) in {PLOT_DIR}")

# ---------------- Plotting ----------------
def plot_barh(df, title, outfile):
    means = df["mean"].values
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(y, means)
    if df["has_ci"].iloc[0]:
        lo = means - df["ci_lo"].values
        hi = df["ci_hi"].values - means
        ax.errorbar(means, y, xerr=[lo, hi], fmt="none", capsize=3, linewidth=1, color="black")
    ax.axvline(0, linestyle="--", linewidth=1, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature_pretty"].values)
    ax.invert_yaxis()
    ax.set_title(title)
    # significance
    for yi, xi, p in zip(y, means, df["p_adj"].values):
        if np.isfinite(p) and p < 0.05:
            ax.text(xi, yi, " *", va="center", ha="left", fontsize=10)
    plt.tight_layout()
    outpath = os.path.join(PLOT_DIR, outfile) if not outfile.startswith(PLOT_DIR) else outfile
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {os.path.relpath(outpath)}")

def plot_grid(stats_by_exp, title, outfile):
    exps = list(stats_by_exp.keys())
    if len(exps) == 0:
        raise RuntimeError("plot_grid received 0 experiments")

    ncols = 3
    nrows = int(np.ceil(len(exps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False, sharex=True)

    # Common x-limits across panels
    mins, maxs = [], []
    for df in stats_by_exp.values():
        if df["has_ci"].iloc[0]:
            mins.append(np.nanmin(df["ci_lo"]))
            maxs.append(np.nanmax(df["ci_hi"]))
        else:
            mins.append(np.nanmin(df["mean"]))
            maxs.append(np.nanmax(df["mean"]))
    xlim = (np.nanmin(mins), np.nanmax(maxs))

    for i, exp in enumerate(exps):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        df = stats_by_exp[exp]
        means = df["mean"].values
        y = np.arange(len(df))
        ax.barh(y, means)
        if df["has_ci"].iloc[0]:
            lo = means - df["ci_lo"].values
            hi = df["ci_hi"].values - means
            ax.errorbar(means, y, xerr=[lo, hi], fmt="none", capsize=3, linewidth=1, color="black")
        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(df["feature_pretty"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(EXP_FULLNAME_MAP.get(exp, exp), fontsize=11)
        for yi, xi, p in zip(y, means, df["p_adj"].values):
            if np.isfinite(p) and p < 0.05:
                ax.text(xi, yi, " *", va="center", ha="left", fontsize=8)
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.set_xlabel("Importance (mean ± 95% CI)" if df["has_ci"].iloc[0] else "Importance (mean)")

    # Hide unused axes
    for j in range(len(exps), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = outfile
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {os.path.relpath(outpath)}")


def plot_double(df_left, df_right, labels=("Coefficients", "Permutation"),
                title=None, outfile=None, sort_each=True):
    """
    Side-by-side comparison with fully independent subplots.
    - No shared axes.
    - No feature alignment between panels.
    - Each panel can be sorted independently by its own mean (descending).
    """
    # Optionally sort each panel independently by mean (descending)
    L = df_left.sort_values("mean", ascending=False).reset_index(drop=True) if sort_each else df_left.copy()
    R = df_right.sort_values("mean", ascending=False).reset_index(drop=True) if sort_each else df_right.copy()

    # Independent y positions
    yL = np.arange(len(L))
    yR = np.arange(len(R))

    fig, axes = plt.subplots(ncols=2, figsize=(10, 6))  # no sharey

    # --- Left panel ---
    ax = axes[0]
    m = L["mean"].values
    ax.barh(yL, m)
    if bool(L["has_ci"].iloc[0]):
        lo = m - L["ci_lo"].values
        hi = L["ci_hi"].values - m
        ax.errorbar(m, yL, xerr=[lo, hi], fmt="none", capsize=3, linewidth=1, color="black")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(yL)
    ax.set_yticklabels(L["feature_pretty"].values)
    ax.invert_yaxis()
    ax.set_title(labels[0])
    if "p_adj" in L.columns:
        for yi, xi, p in zip(yL, m, L["p_adj"].values):
            if np.isfinite(p) and p < 0.05:
                ax.text(xi, yi, " *", va="center", ha="left", fontsize=9)

    # --- Right panel ---
    ax = axes[1]
    m = R["mean"].values
    ax.barh(yR, m)
    if bool(R["has_ci"].iloc[0]):
        lo = m - R["ci_lo"].values
        hi = R["ci_hi"].values - m
        ax.errorbar(m, yR, xerr=[lo, hi], fmt="none", capsize=3, linewidth=1, color="black")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(yR)
    ax.set_yticklabels(R["feature_pretty"].values)
    ax.invert_yaxis()
    ax.set_title(labels[1])
    if "p_adj" in R.columns:
        for yi, xi, p in zip(yR, m, R["p_adj"].values):
            if np.isfinite(p) and p < 0.05:
                ax.text(xi, yi, " *", va="center", ha="left", fontsize=9)

    # Overall title + save
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if not outfile:
        raise ValueError("outfile is required for plot_double")
    outpath =outfile
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {os.path.relpath(outpath)}")

    
# ---------------- Main orchestration ----------------
def main():
    if not os.path.isdir(PLOT_DIR):
        raise FileNotFoundError(f"PLOT_DIR not found: {PLOT_DIR}")

    # Discover experiments present on disk
    files = [f for f in os.listdir(PLOT_DIR) if f.endswith(".csv")]
    found = sorted(set(re.sub(r"_LogReg.*", "", f) for f in files))
    # Restrict to canonical 7, preserve order
    exps = [e for e in DESIRED_ORDER if e in found]
    print("[discover] found:", found)
    print("[discover] will plot:", exps)
    if len(exps) == 0:
        raise RuntimeError("No canonical experiments found in plots directory.")

    # Build grids (weights and permutation) for the 7 only
    stats_w = {e: load_stats(e, "weights") for e in exps}
    plot_grid(stats_w,
              "Logistic Regression Feature Weights (mean ± 95% CI)",
              "feature_weights_grid_CI.png")

    stats_p = {e: load_stats(e, "perm") for e in exps}
    plot_grid(stats_p,
              "Logistic Regression Permutation Importances (mean ± 95% CI)",
              "perm_importances_grid_CI.png")

    # Combined side-by-side figure for the 3-way combo (if present)
    combo = "clinical parameters_pleomorphism_densities"
    if combo in exps:
        Sw = stats_w[combo]
        Sp = stats_p[combo]
        plot_double(Sw, Sp,
                    labels=("Coefficients", "Permutation"),
                    title=EXP_FULLNAME_MAP.get(combo, combo),
                    outfile=f"{combo}_weights_vs_perm_CI.png")

if __name__ == "__main__":
    main()
