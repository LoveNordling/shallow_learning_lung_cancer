#!/usr/bin/env python3
import os
import re
import csv
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score

import forestplot as fp
from forestplot_custom import simple_forestplot


# ---------- Helpers ----------

def clean_model_label(s: str) -> str:
    """For display: remove trailing ' logit' and ' pred'/' pred.'; keep underscores."""
    return (
        s.replace(" logit", "")
         .replace(" pred.", "")
         .replace(" pred", "")
    )

def median_split_strict(df: pd.DataFrame, col: str) -> pd.Series:
    """Binary group by value-based median with a strict '>' rule."""
    x = df[col].astype(float)
    if x.nunique() < 2:
        return pd.Series(np.nan, index=df.index, name=f"{col}_bin")
    cutoff = x.median()
    grp = (x > cutoff).astype(int)
    grp.name = f"{col}_bin"
    return grp

def dichotomize_clinpars(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Age"] = data["Age"].map(lambda x: 1 if x > 70 else 0)
    data["Performance status"] = data["Performance status"].map(lambda x: 1 if x > 0 else 0)
    data["Stage"] = data["Stage"].map(lambda x: 1 if x > 1 else 0)
    return data

def cencor_after5years(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.loc[data["time"] > 5 * 365, "event"] = 0
    return data

def get_best_cutoff(data: pd.DataFrame, feature: str):
    """Optional utility: scan percentiles 10..89 and pick best log-rank split."""
    df = data.copy()
    best_p_value = 10
    best_cutoff = None
    for cutoff in np.percentile(df[feature], np.arange(10, 90, 1)):
        df["group"] = (df[feature] > cutoff).astype(int)
        if df["group"].nunique() < 2:
            continue
        results = logrank_test(
            df['time'][df['group'] == 0], df['time'][df['group'] == 1],
            event_observed_A=df['event'][df['group'] == 0],
            event_observed_B=df['event'][df['group'] == 1],
        )
        if results.p_value < best_p_value:
            best_p_value = results.p_value
            best_cutoff = cutoff
    print(feature, best_p_value)
    return best_cutoff


def get_all_best_cutoffs(data):
    """
    Scan non-model, non-clinical numeric features and find the cutoff (10th–90th pct)
    that gives the best log-rank separation. Returns {feature -> cutoff}.
    """
    df = data.copy()
    # columns to skip outright
    exclude = {
        "ID", "label", "time", "event",
        "LUAD", "Age", "Sex (Male)", "Smoking", "Stage", "Performance status",
    }
    # candidate columns: numeric, not excluded, not model outputs
    candidates = [
        c for c in df.columns
        if c not in exclude
        and not c.endswith((" pred", " pred.", " logit"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    cutoffs = {}
    for col in tqdm(candidates):
        x = df[col].dropna()
        # skip constants or already-binary
        if x.nunique() < 3 or set(x.unique()).issubset({0, 1}):
            continue
        cutoff = get_best_cutoff(df[["time", "event", col]].dropna(), col)
        if cutoff is not None:
            cutoffs[col] = cutoff
    return cutoffs


# ---------- Plotting ----------

def plot_kaplan_meier_subplots(
    data, time_col, event_col, feature_cols, fig_title, plot_names, ncols=3, out_path=None, group_labels_dict=None
):
    import math
    if not feature_cols:
        print("KM subplots: no features to plot — skipping.")
        return
    n_plots = len(feature_cols)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, (feature_col, plot_name) in enumerate(zip(feature_cols, plot_names)):
        ax = axes[idx]
        kmf = KaplanMeierFitter()

        # Plot 0/1 groups
        for value in [0, 1]:
            subset = data[data[feature_col] == value]
            if len(subset) == 0:
                continue
            label = (group_labels_dict.get(feature_col, {0: "Short", 1: "Long"}).get(value, str(value))
                     if group_labels_dict else ("Long" if value == 1 else "Short"))
            kmf.fit(subset[time_col], event_observed=subset[event_col], label=label)
            kmf.plot_survival_function(ci_show=True, ax=ax)

        # Log-rank test instead of fraction text
        g0 = data[data[feature_col] == 0]
        g1 = data[data[feature_col] == 1]
        if len(g0) > 0 and len(g1) > 0:
            res = logrank_test(
                g0[time_col], g1[time_col],
                event_observed_A=g0[event_col],
                event_observed_B=g1[event_col]
            )
            ax.text(0.95, 0.05, f"Log-rank p = {res.p_value:.3g}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=15, bbox=dict(facecolor='white', alpha=0.5))

        ax.set_title(plot_name, fontsize=16)
        ax.set_xlabel("Overall survival time (Years)", fontsize=13)
        ax.set_ylabel("Survival probability", fontsize=13)
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5)
        ax.legend()

    # Hide unused axes and save
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_model_kaplan_meier(data, time_col, event_col, model_cols, fig_title, out_path=None):
    import math
    model_cols = list(model_cols)
    if not model_cols:
        print("KM models: no model columns to plot — skipping.")
        return
    n_plots = len(model_cols)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, col in enumerate(model_cols):
        ax = axes[idx]

        # Use pre-dichotomized column if already binary; otherwise median split
        if set(pd.unique(data[col]).astype(float)) <= {0.0, 1.0}:
            bin_col = col
        else:
            bin_col = f"{col}_bin"
            data[bin_col] = (data[col] > data[col].median()).astype(int)

        # Plot 0/1 groups
        kmf = KaplanMeierFitter()
        for value in [0, 1]:
            subset = data[data[bin_col] == value]
            if len(subset) == 0:
                continue
            kmf.fit(subset[time_col], event_observed=subset[event_col],
                    label="Long" if value == 1 else "Short")
            kmf.plot_survival_function(ci_show=True, ax=ax)

        # Log-rank test instead of fraction text
        g0 = data[data[bin_col] == 0]
        g1 = data[data[bin_col] == 1]
        if len(g0) > 0 and len(g1) > 0:
            res = logrank_test(
                g0[time_col], g1[time_col],
                event_observed_A=g0[event_col],
                event_observed_B=g1[event_col]
            )
            ax.text(0.95, 0.05, f"Log-rank p = {res.p_value:.3g}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=15, bbox=dict(facecolor='white', alpha=0.5))

        # Bigger titles and “_” → “+” for model names
        ax.set_title(clean_model_label(col).replace("_", "+"), fontsize=16)
        ax.set_xlabel("Overall survival time (Years)", fontsize=13)
        ax.set_ylabel("Survival probability", fontsize=13)
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5)
        ax.legend()

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)
    if out_path:
        print("Saved KM to", out_path)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")


# ---------- Cox ----------

def multivariate_coxregression(data, p_values):
    data = data.drop(columns=["ID", "label"], errors="ignore")
    columns = [
        x for x in p_values
        if (p_values[x] < 0.05)
        and (not x.endswith(" pred"))
        and (not x.endswith(" pred."))
        and (not x.endswith(" logit"))
        and (x not in ["time", "event"])
        and (x in data.columns)
    ]
    if not columns:
        print("Multivariate Cox: no eligible features after filtering — skipping.")
        return
    df = data[["time", "event"] + columns].copy()
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    summary = cph.summary.reset_index()
    summary = summary[["covariate", "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    summary.columns = ["Feature", "Coefficient", "Hazard Ratio", "Lower CI", "Upper CI", "p"]

    fig = simple_forestplot(summary, est_col="Hazard Ratio",
                            ll_col="Lower CI", hl_col="Upper CI",
                            feature_col="Feature", pval_col="p",
                            xlabel="Hazard ratio")
    plt.savefig("multivariate_cox.png", dpi=300, bbox_inches="tight")

def univariate_coxregression(data, model_logit_cols, out_features="univariate_cox_features.png",
                             out_models="univariate_cox_models_logits.png"):
    """
    Runs univariate Cox on every column except ['time','event'].
    Produces two figures:
      1) features (non-model outputs),
      2) model logits (ordered to match KM), labels without 'logit'.
    Returns: dict of p-values by original column name.
    """
    df = data.drop(columns=["ID", "label"], errors="ignore").copy()
    cph = CoxPHFitter()

    results = []
    p_values = {}

    for feature in df.columns:
        if feature in ("time", "event"):
            continue
        # Skip completely-constant columns
        if df[feature].nunique(dropna=False) <= 1:
            continue
        try:
            cph.fit(df[["time", "event", feature]], duration_col="time", event_col="event")
            s = cph.summary.loc[feature, ['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
            results.append([feature, s['coef'], s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']])
            p_values[feature] = float(s['p'])
        except Exception:
            # Some columns may fail numerically
            continue

    if not results:
        print("Univariate Cox: no results — skipping plots.")
        return {}

    summary_df = pd.DataFrame(results, columns=['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', 'p-value'])

    # 1) Non-model features (exclude all model outputs)
    feat_mask = ~summary_df["Feature"].str.endswith((" pred", " pred.", " logit"), na=False)
    summary_features = summary_df.loc[feat_mask].copy()
    if not summary_features.empty:
        fig = simple_forestplot(summary_features, est_col="Hazard Ratio",
                                ll_col="Lower CI", hl_col="Upper CI",
                                feature_col="Feature", pval_col="p-value",
                                xlabel="Hazard ratio")
        plt.savefig(out_features, dpi=300, bbox_inches="tight")

    # 2) Model logits only, ordered like KM; strip ' logit' in labels
    logits_mask = summary_df["Feature"].str.endswith(" logit", na=False)
    summary_logits = summary_df.loc[logits_mask].copy()
    if not summary_logits.empty:
        # Desired order = the order of the columns we plot in KM
        """model_order_clean = [clean_model_label(c) for c in model_logit_cols]
        summary_logits["Feature"] = summary_logits["Feature"].map(clean_model_label)
        # Reindex to that order, keeping only those present
        cats = [m for m in model_order_clean if m in set(summary_logits["Feature"])]
        summary_logits["Feature"] = pd.Categorical(summary_logits["Feature"], categories=cats, ordered=True)
        summary_logits = summary_logits.sort_values("Feature")"""
        # Clean labels for display and switch "_" → "+" so they match the paper style
        summary_logits["Feature"] = (
            summary_logits["Feature"]
            .map(clean_model_label)
            .str.replace("_", "+", regex=False)
        )

        # Order to match the KM models (also using "+" for consistency)
        model_order_plus = [clean_model_label(c).replace("_", "+") for c in model_logit_cols]
        cats = [m for m in model_order_plus if m in set(summary_logits["Feature"])]
        summary_logits["Feature"] = pd.Categorical(summary_logits["Feature"], categories=cats, ordered=True)
        summary_logits = summary_logits.sort_values("Feature")

        fig = simple_forestplot(summary_logits, est_col="Hazard Ratio",
                                ll_col="Lower CI", hl_col="Upper CI",
                                feature_col="Feature", pval_col="p-value",
                                xlabel="Hazard ratio")
        plt.savefig(out_models, dpi=300, bbox_inches="tight")

    return p_values


# ---------- Data IO for patient-level predictions ----------

def get_patient_preds():
    """
    Reads averaged per-patient predictions for each experiment:
      ./preds/{experiment}_LogReg_predictions.csv
      ./preds/{experiment}_LogReg_logits.csv
    Returns: (experiment_names, merged_dataframe)
    """
    names = [x.replace("_LogReg_predictions.csv", "")
             for x in os.listdir("./preds/")
             if x.endswith("_LogReg_predictions.csv")]

    all_model_preds = []
    for experiment_name in names:
        preds = pd.read_csv(os.path.join("./preds/", f"{experiment_name}_LogReg_predictions.csv"))
        logits = pd.read_csv(os.path.join("./preds/", f"{experiment_name}_LogReg_logits.csv"))

        preds = preds.mean(axis=0).reset_index()
        preds.columns = ["ID", f"{experiment_name} pred"]

        logits = logits.mean(axis=0).reset_index()
        logits.columns = ["ID", f"{experiment_name} logit"]

        patient_preds = pd.merge(logits, preds, on="ID")
        all_model_preds.append(patient_preds)

    all_model_preds = reduce(lambda left, right: pd.merge(left, right, on='ID'), all_model_preds)
    return names, all_model_preds


# ---------- Main pipeline ----------

if __name__ == "__main__":
    print("get patient predictions")
    experiment_names, all_models_preds = get_patient_preds()

    patient_data = pd.read_csv("patient_densities_morphologies.csv")
    patient_data.rename(columns={"Gender": "Sex (Male)"}, inplace=True)
    patient_data['ID'] = patient_data['ID'].astype(float)

    all_models_preds['ID'] = all_models_preds['ID'].astype(float)

    # Clinical preprocessing
    patient_data = dichotomize_clinpars(patient_data)
    patient_data = cencor_after5years(patient_data)

    # Merge predictions
    patient_data_model_preds = pd.merge(patient_data, all_models_preds, on="ID")
    patient_data = patient_data_model_preds.copy()

    # Optional quick sanity metric if available
    if "clinical parameters pred" in patient_data_model_preds.columns:
        try:
            auc = roc_auc_score(patient_data_model_preds["label"], patient_data_model_preds["clinical parameters pred"])
            print("accuracy (AUC, clinical parameters pred):", auc)
        except Exception as e:
            print("AUC calc skipped:", e)

    # Rename density feature labels for readability (non-model features)
    density_features = [
        "Stroma CD4_Single", "Stroma CD4_Treg", "Stroma CD8_Single", "Stroma CD8_Treg", "Stroma B_cells",
        "Tumor CD4_Single", "Tumor CD4_Treg", "Tumor CD8_Single", "Tumor CD8_Treg", "Tumor B_cells"
    ]
    density_rename_map = {}
    for col in density_features:
        if col in patient_data.columns:
            new_col = col.replace("_", " ").replace("Single", "eff.")
            density_rename_map[col] = new_col
            patient_data.rename(columns={col: new_col}, inplace=True)
    density_features = [density_rename_map.get(col, col) for col in density_features]

    pleomorphism_features = ['Nucleus Area', 'Nucleus Compactness', 'Nucleus Axis Ratio']
    clinical_features = ['Age', 'Performance status', 'Stage', 'Sex (Male)', 'Smoking', 'LUAD']
    clinical_titles = clinical_features  # same names in KM grids
    density_titles = density_features
    pleomorphism_titles = ["Nucleus area", "Nucleus compactness", "Nucleus axis ratio"]

    # Models: use logits only; build a canonical order with underscores
    MODEL_NAME_ORDER = [
        "clinical parameters",
        "densities",
        "pleomorphism",
        "clinical parameters_densities",
        "clinical parameters_pleomorphism",
        "pleomorphism_densities",
        "clinical parameters_pleomorphism_densities",
    ]
    # pick the logits that exist, in that order
    model_logit_cols = [f"{m} logit" for m in MODEL_NAME_ORDER if f"{m} logit" in patient_data.columns]

    # Dichotomize model logits by strict median
    for col in model_logit_cols:
        patient_data[col] = median_split_strict(patient_data, col)

    # KM for clinical/density/pleomorphism (binary vars)
    patient_data["time"] = patient_data["time"] / 365.0  # days → years

    clinical_group_labels = {
        "Age": {0: "<70", 1: "70+"},
        "Stage": {0: "Stage I", 1: "Stage II–IV"},
        "Performance status": {0: "0", 1: "1–2"},
        "Sex (Male)": {0: "Female", 1: "Male"},
        "Smoking": {0: "Never", 1: "Ever"},
        "LUAD": {0: "Other/SqCC", 1: "LUAD"},
    }

    #Dichotomize continuous non-model features by optimal log-rank cutoffs
    cutoffs = get_all_best_cutoffs(patient_data)
    for feat, cutoff in cutoffs.items():
        patient_data[feat] = (patient_data[feat] > cutoff).astype(int)

    plot_kaplan_meier_subplots(
        patient_data, "time", "event", clinical_features, "Kaplan-Meier: Clinical Parameters",
        clinical_titles, ncols=3, out_path="km_clinical.png", group_labels_dict=clinical_group_labels
    )
    plot_kaplan_meier_subplots(
        patient_data, "time", "event", density_features, "Kaplan-Meier: Cell Densities",
        density_titles, ncols=3, out_path="km_densities.png"
    )
    plot_kaplan_meier_subplots(
        patient_data, "time", "event", pleomorphism_features, "Kaplan-Meier: Pleomorphism",
        pleomorphism_titles, ncols=3, out_path="km_pleomorphism.png"
    )

    # KM for models: logits only; titles without 'logit'
    plot_model_kaplan_meier(
        patient_data, "time", "event", model_logit_cols, "Kaplan–Meier: Model Logits",
        out_path="km_models_logit.png"
    )

    # --- KM: main models (logits only) ---
    main_keep = {"clinical parameters", "densities", "clinical parameters_densities"}
    main_model_logit_cols = [c for c in model_logit_cols if clean_model_label(c) in main_keep]

    plot_model_kaplan_meier(
        patient_data, "time", "event",
        main_model_logit_cols,
        "Kaplan–Meier: Main Models (Logits)",
        out_path="km_main_models_logit.png"
    )

    # Univariate Cox: features + model logits (ordered same as KM), labels without 'logit'
    p_values = univariate_coxregression(
        patient_data,
        model_logit_cols=model_logit_cols,
        out_features="univariate_cox_features.png",
        out_models="univariate_cox_models_logits.png",
    )

    # Multivariate + stepwise (exclude any model outputs)
    multivariate_coxregression(patient_data, p_values)
    # If you want stepwise variants back, re-add them here (they currently filter out ' logit'/' pred').
