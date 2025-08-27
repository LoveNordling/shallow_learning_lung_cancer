import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_rel, shapiro, wilcoxon
import matplotlib.pyplot as plt
from tqdm import tqdm

# Toggle permutation importances
PERMUTE_FEATURES = True

def _p_one_sided(x, y, label, alpha=0.05):
    """One-sided paired test for 'is x > y?':
       - Shapiro on paired diffs
       - If non-normal -> Wilcoxon (greater)
       - Else -> paired t-test (greater)
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size == 0:
        return np.nan

    diffs = x - y

    # Shapiro on paired differences (relevant assumption for paired t-test)
    try:
        p_norm = shapiro(diffs).pvalue if 3 < diffs.size < 5000 else 1.0
    except Exception:
        p_norm = 1.0

    if p_norm < alpha:
        print(f"[Warning] Non-normal paired differences for {label} (Shapiro p={p_norm:.3g}); using Wilcoxon (one-sided).")
        try:
            return wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue
        except ValueError:
            # All diffs zero or too few non-zero pairs – no evidence of improvement
            return 1.0
    else:
        return ttest_rel(x, y, alternative="greater").pvalue


def get_feature_importance(model, columns):
    if hasattr(model, 'coef_'):
        return pd.DataFrame({'feature': columns, 'importance': model.coef_[0]})
    elif hasattr(model, 'feature_importances_'):
        return pd.DataFrame({'feature': columns, 'importance': model.feature_importances_})
    else:
        return pd.DataFrame({'feature': columns, 'importance': [np.nan]*len(columns)})
    
def plot_feature_importances(importances, feature_names, title, filename):
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class RandomClassifier:
    def fit(self, X, y):
        self.label_proportions = y.value_counts(normalize=True)
    def predict(self, X):
        return np.random.choice(self.label_proportions.index, size=len(X), p=self.label_proportions.values)
    def predict_proba(self, X):
        return np.tile(self.label_proportions.values, (len(X), 1))

def init_rf():
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    return GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_logreg():
    param_grid = {
        'C': np.logspace(-4, 4, 10),#[0.1, 1, 10],#(-4, 4, 10),  # Try a wide range of regularization strengths
        'penalty': ['l1', 'l2'],      # Test both L1 and L2 regularization
        'solver': ['liblinear', 'saga'],  # Solvers that support L1 and L2 penalties
        'max_iter': [10000]   # Ensure enough iterations for convergence
    }
    log_reg = LogisticRegression(random_state=42)
    return GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_svm():
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }
    svm = SVC(probability=True, random_state=42)
    return GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_knn():
    param_grid = {'n_neighbors': [5, 10, 20]}
    knn = KNeighborsClassifier()
    return GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

model_registry = {
    "RF": init_rf,
    "LogReg": init_logreg,
    "SVM": init_svm,
    "KNN": init_knn,
    "Random": lambda: RandomClassifier()
}

# ---- Main ----

df = pd.read_csv("patient_densities_morphologies.csv")
X = df.drop(columns=["ID", 'time', 'event', 'label'])
y = df[["ID", "label"]]

column_names = X.columns.tolist()
clinical_parameters = column_names[:6]
pleomorphism = column_names[6:9]
densities = column_names[9:]
part_names = {"clinical parameters": clinical_parameters, "pleomorphism": pleomorphism, "densities": densities}

name_combination_list = [
    ["clinical parameters","pleomorphism","densities"],
    ["clinical parameters","pleomorphism"],
    ["clinical parameters","densities"],
    ["pleomorphism","densities"],
    ["clinical parameters"],
    ["pleomorphism"],
    ["densities"]
]

split_folder = "/data2/love/multiplex_cancer_cohorts/patient_and_samples_data/lung_cancer_BOMI2_dataset/binary_survival_prediction/100foldcrossvalrepeat/"
num_splits = len([x for x in os.listdir(split_folder) if "test" in x])

os.makedirs("plots", exist_ok=True)
os.makedirs("preds", exist_ok=True)

results_dict_titles = [
    "Experiment", "Model",
    "Train accuracy mean", "Train accuracy std", "Train AUC mean", "Train AUC std",
    "Test accuracy mean", "Test accuracy std", "Test accuracy sem",
    "Test AUC mean", "Test AUC std", "Test AUC sem",
    "Test sensitivity", "Test specificity", "Test NPV", "Test PPV"
]
results_dict = {t: [] for t in results_dict_titles}

experiment_results_acc = {}
experiment_results_auc = {}

# For p-value stats
all_acc = {}
all_auc = {}

for name_comb in name_combination_list:
    experiment_name = "_".join(name_comb)
    feature_list = sum([part_names[part] for part in name_comb], [])
    print(f"=== Experiment: {experiment_name} | Features: {feature_list}")

    X_experiment = X[feature_list]
    for model_name, model_fn in model_registry.items():
        # Storage for per-split
        test_accs, test_aucs, test_sens, test_spec, test_npv, test_ppv = [], [], [], [], [], []
        train_accs, train_aucs = [], []
        feature_importances = []
        permutation_importances = []
        preds_by_id = {ID: [] for ID in df["ID"]}
        logits_by_id = {ID: [] for ID in df["ID"]}
        preds_logits = []
        fi_per_split_rows = []   # coefficients (LogReg)
        pi_per_split_rows = []   # permutation importance
        
        for split in tqdm(range(num_splits), desc=f"{experiment_name}-{model_name}", leave=False):
            train_ids = pd.read_csv(os.path.join(split_folder, f"split_{split}_train_val.csv"))["ID"]
            test_ids = pd.read_csv(os.path.join(split_folder, f"split_{split}_test.csv"))["ID"]
            train_mask = df["ID"].isin(train_ids)
            test_mask = df["ID"].isin(test_ids)
            X_train, X_test = X_experiment[train_mask], X_experiment[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = model_fn()
            # Fit
            if model_name == "Random":
                model.fit(X_train_scaled, y_train["label"])
                preds = model.predict(X_test_scaled)
                probas = model.predict_proba(X_test_scaled)[:, 1]
                preds_train = model.predict(X_train_scaled)
                probas_train = model.predict_proba(X_train_scaled)[:, 1]
                best_model = model
            else:
                model.fit(X_train_scaled, y_train["label"])
                best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
                preds = best_model.predict(X_test_scaled)
                probas = best_model.predict_proba(X_test_scaled)[:, 1]
                preds_train = best_model.predict(X_train_scaled)
                probas_train = best_model.predict_proba(X_train_scaled)[:, 1]

            # Metrics
            acc = accuracy_score(y_test["label"], preds)
            auc = roc_auc_score(y_test["label"], probas)
            acc_train = accuracy_score(y_train["label"], preds_train)
            auc_train = roc_auc_score(y_train["label"], probas_train)
            test_accs.append(acc)
            test_aucs.append(auc)
            train_accs.append(acc_train)
            train_aucs.append(auc_train)

            cm = confusion_matrix(y_test["label"], preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            sensitivity = recall_score(y_test["label"], preds) if (tp+fn) else 0
            specificity = tn / (tn+fp) if (tn+fp) else 0
            ppv = precision_score(y_test["label"], preds) if (tp+fp) else 0
            npv = tn / (tn+fn) if (tn+fn) else 0
            test_sens.append(sensitivity)
            test_spec.append(specificity)
            test_npv.append(npv)
            test_ppv.append(ppv)

            # Store per-patient results
            test_ids_vals = df[test_mask]["ID"].values
            preds_logits.append(pd.DataFrame({
                "ID": test_ids_vals,
                "split": split,
                "prediction": preds,
                "logit": probas
            }))
            for ID, pred, logit in zip(test_ids_vals, preds, probas):
                preds_by_id[ID].append(pred)
                logits_by_id[ID].append(logit)

            # Feature importances
            if model_name in ["LogReg"]:
                fi = get_feature_importance(best_model, feature_list)
                feature_importances.append(fi)
                # After computing 'fi' (a DataFrame with columns ['feature','importance'])
                for f, v in zip(fi["feature"].values, fi["importance"].values):
                    fi_per_split_rows.append({
                        "split": split,           # your fold/repeat index
                        "feature": f,
                        "importance": float(v)
                    })
                if PERMUTE_FEATURES:
                    pi = permutation_importance(best_model, X_test_scaled, y_test["label"], n_repeats=10, random_state=42, n_jobs=-1)#, scoring="roc_auc")
                    permutation_importances.append(pi.importances_mean)
                    # Save per-fold permutation importances (mean across repeats)
                    for f, v in zip(feature_list, pi.importances_mean):
                        pi_per_split_rows.append({
                            "split": split,
                            "feature": f,
                            "importance": float(v)
                        })

        # Write raw per-fold explainability (only if collected)
        if model_name == "LogReg" and len(fi_per_split_rows) > 0:
            pd.DataFrame(fi_per_split_rows).to_csv(
                f"plots/{experiment_name}_{model_name}_feature_importance_per_split.csv",
                index=False
            )
        if len(pi_per_split_rows) > 0:
            pd.DataFrame(pi_per_split_rows).to_csv(
                f"plots/{experiment_name}_{model_name}_perm_importance_per_split.csv",
                index=False
            )
        # Aggregate results for results_dict
        results_dict["Experiment"].append(experiment_name)
        results_dict["Model"].append(model_name)
        results_dict["Train accuracy mean"].append(np.mean(train_accs))
        results_dict["Train accuracy std"].append(np.std(train_accs))
        results_dict["Train AUC mean"].append(np.mean(train_aucs))
        results_dict["Train AUC std"].append(np.std(train_aucs))
        results_dict["Test accuracy mean"].append(np.mean(test_accs))
        results_dict["Test accuracy std"].append(np.std(test_accs))
        results_dict["Test accuracy sem"].append(np.std(test_accs)/np.sqrt(len(test_accs)))
        results_dict["Test AUC mean"].append(np.mean(test_aucs))
        results_dict["Test AUC std"].append(np.std(test_aucs))
        results_dict["Test AUC sem"].append(np.std(test_aucs)/np.sqrt(len(test_aucs)))
        results_dict["Test sensitivity"].append(np.mean(test_sens))
        results_dict["Test specificity"].append(np.mean(test_spec))
        results_dict["Test NPV"].append(np.mean(test_npv))
        results_dict["Test PPV"].append(np.mean(test_ppv))

        # Store for stats
        all_acc[(experiment_name, model_name)] = test_accs
        all_auc[(experiment_name, model_name)] = test_aucs

        # Per-ID, split predictions/logits
        predictions_df = pd.DataFrame(preds_by_id)
        logits_df = pd.DataFrame(logits_by_id)
        preds_logits_cat = pd.concat(preds_logits, ignore_index=True)
        preds_logits_grouped = preds_logits_cat.groupby("ID").agg({
            "prediction": "mean",
            "logit": "mean"
        }).rename(columns={"prediction": "mean_prediction", "logit": "mean_logit"})
        preds_logits_grouped.to_csv(f"preds/{experiment_name}_{model_name}_preds_logits.csv", index=False)
        predictions_df.to_csv(f"preds/{experiment_name}_{model_name}_predictions.csv", index=False)
        logits_df.to_csv(f"preds/{experiment_name}_{model_name}_logits.csv", index=False)

        # Feature importances and plots
        if feature_importances:
            fi_cat = pd.concat(feature_importances)
            avg_fi = fi_cat.groupby('feature')['importance'].mean().reset_index()
            avg_fi = avg_fi.sort_values(by='importance', ascending=False)
            avg_fi.to_csv(f"plots/{experiment_name}_{model_name}_feature_importance.csv", index=False)
            plot_feature_importances(avg_fi['importance'].values, avg_fi['feature'].values,
                f"Feature Importances: {experiment_name} {model_name}",
                f"plots/{experiment_name}_{model_name}_feature_importance.png")
        # Permutation importance (if enabled)
        if PERMUTE_FEATURES and permutation_importances:
            avg_perm = np.mean(permutation_importances, axis=0)
            #np.save(f"plots/{experiment_name}_{model_name}_perm_importance.npy", avg_perm)
            # Save permutation importance as CSV with feature names
            perm_df = pd.DataFrame({'feature': feature_list, 'importance': avg_perm})
            perm_df.to_csv(f"plots/{experiment_name}_{model_name}_perm_importance.csv", index=False)

            # Optionally plot
            plot_feature_importances(avg_perm, feature_list,
                f"Permutation Importances: {experiment_name} {model_name}",
                f"plots/{experiment_name}_{model_name}_perm_importance.png")

    # Save for baseline comparison (for stats)
    experiment_results_acc[experiment_name] = all_acc.get((experiment_name, "LogReg"), [])
    experiment_results_auc[experiment_name] = all_auc.get((experiment_name, "LogReg"), [])

# Save results to CSV
results = pd.DataFrame(results_dict)
results.to_csv("results_shallow_learning2_full.csv", index=False)

# Paired t-test vs clinical baseline (only for accuracy/AUC)
p_value_stats = {"experiment": [], "model": [], "Accuracy": [], "AUC": []}

for key in all_acc:
    experiment_name, model_name = key
    clinical_acc = all_acc.get(("clinical parameters", model_name))
    clinical_auc = all_auc.get(("clinical parameters", model_name))
    if experiment_name == "clinical parameters":# and model_name == "LogReg":
        continue
    results_acc = all_acc[key]
    results_auc = all_auc[key]
    if clinical_acc is not None and len(results_acc) == len(clinical_acc):
        #_, p_value_acc = ttest_rel(results_acc, clinical_acc)
        #_, p_value_auc = ttest_rel(results_auc, clinical_auc)
        p_value_acc = _p_one_sided(results_acc, clinical_acc,
                                          f"{experiment_name} | {model_name} | Accuracy")
        p_value_auc = _p_one_sided(results_auc, clinical_auc,
                                          f"{experiment_name} | {model_name} | AUC")


        p_value_stats["experiment"].append(experiment_name)
        p_value_stats["model"].append(model_name)
        p_value_stats["Accuracy"].append(p_value_acc)
        p_value_stats["AUC"].append(p_value_auc)
pd.DataFrame(p_value_stats).to_csv("p_values.csv", index=False)



# ---- Compact, ordered results per model ----
# Uses test-set metrics only, ordered for figures/tables, with human-readable names.

# 1) Human-readable names and desired order (to match your plotting scripts)
exp_fullname_map = {
    "clinical parameters": "Clinical Parameters",
    "densities": "Immune Cell Densities",
    "pleomorphism": "Pleomorphism",
    "clinical parameters_densities": "Clinical Parameters + Densities",
    "clinical parameters_pleomorphism": "Clinical Parameters + Pleomorphism",
    "pleomorphism_densities": "Densities + Pleomorphism",
    "clinical parameters_pleomorphism_densities": "Clinical Parameters + Densities + Pleomorphism",
}
desired_order = [
    "clinical parameters",
    "densities",
    "pleomorphism",
    "clinical parameters_densities",
    "clinical parameters_pleomorphism",
    "pleomorphism_densities",
    "clinical parameters_pleomorphism_densities",
]

# 2) Helpers
def _fmt_mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    sem = np.std(arr, ddof=0) / np.sqrt(len(arr)) if len(arr) else np.nan
    return f"{mean:.3f} ± {sem:.3f}"

def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

# Convert the long 'results' table to a dict for quick lookups (means of other test metrics)
_results_df = results  # from earlier in the script
_results_idx = {(r["Experiment"], r["Model"]): r for _, r in _results_df.iterrows()}

# 3) Build one compact CSV per model
for model_name in model_registry.keys():
    rows = []
    for exp in desired_order:
        # Skip rows that weren't computed (defensive)
        if (exp, model_name) not in all_acc or (exp, model_name) not in all_auc:
            continue

        # Accuracy mean±SEM and paired t-test vs clinical baseline (same splits)
        acc_vals = all_acc[(exp, model_name)]
        acc_str = _fmt_mean_sem(acc_vals)

        if exp == "clinical parameters":
            p_acc = None
        else:
            base_acc = all_acc.get(("clinical parameters", model_name))
            p_acc = None
            if base_acc is not None and len(base_acc) == len(acc_vals):
                #_, p_acc = ttest_rel(acc_vals, base_acc)
                p_acc = _p_one_sided(acc_vals, base_acc,
                                    f"{exp} | {model_name} | Accuracy")

        # AUC mean±SEM and paired t-test vs clinical baseline
        auc_vals = all_auc[(exp, model_name)]
        auc_str = _fmt_mean_sem(auc_vals)

        if exp == "clinical parameters":
            p_auc = None
        else:
            base_auc = all_auc.get(("clinical parameters", model_name))
            p_auc = None
            if base_auc is not None and len(base_auc) == len(auc_vals):
                #_, p_auc = ttest_rel(auc_vals, base_auc)
                p_auc = _p_one_sided(auc_vals, base_auc,
                                    f"{exp} | {model_name} | AUC")

        # Sens/Spec/NPV/PPV means (3 decimals)
        # (We stored these as means in 'results' earlier)
        rr = _results_idx.get((exp, model_name))
        sens = f"{rr['Test sensitivity']:.3f}" if rr is not None else ""
        spec = f"{rr['Test specificity']:.3f}" if rr is not None else ""
        npv  = f"{rr['Test NPV']:.3f}"         if rr is not None else ""
        ppv  = f"{rr['Test PPV']:.3f}"         if rr is not None else ""

        # Assemble row
        rows.append({
            "Experiment": exp_fullname_map.get(exp, exp),
            "Accuracy": f"{acc_str} ({_fmt_p(p_acc)})",#acc_str,            # mean ± SEM
            #"p-value": _fmt_p(p_acc),       # vs Clinical Parameters
            "AUC": f"{auc_str} ({_fmt_p(p_auc)})",#auc_str,                 # mean ± SEM
            #"p-value.1": _fmt_p(p_auc),     # vs Clinical Parameters
            "Sensitivity": sens,
            "Specificity": spec,
            "NPV": npv,
            "PPV": ppv,
        })

    # Ensure the written order matches desired_order
    compact_cols = ["Experiment", "Accuracy", "p-value", "AUC", "p-value.1", "Sensitivity", "Specificity", "NPV", "PPV"]
    compact_df = pd.DataFrame(rows, columns=compact_cols)

    # Save per model
    compact_out = f"results_shallow_learning2_compact_{model_name}.csv"
    compact_df.to_csv(compact_out, index=False)

