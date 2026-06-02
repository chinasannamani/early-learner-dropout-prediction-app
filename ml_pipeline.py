"""
================================================================================
NUTDTS 803 – Machine Learning Fundamentals
Project : Early Warning ML Model for Prevention of Learner Dropout
          (Nigerian Basic Education – Primary & Secondary Schools)

File    : ml_pipeline.py
Version : 2.0  (v4 dataset — stage-aware real-time early warning)
Author  : [Your Name]
Date    : 2026

Description
-----------
Full ML pipeline covering:
  1. Data Loading & Exploration
  2. Preprocessing (encoding, scaling, SMOTE for class imbalance)
     — v4 additions: Performance_Pct_So_Far NaN fill, Subjects_Failed
       sentinel (-1) kept as meaningful signal, raw unavailable scores
       excluded from features (replaced by engineered stage features)
  3. Model Training  – Logistic Regression, Random Forest, XGBoost
  4. Evaluation      – Accuracy, Precision, Recall, F1, ROC-AUC,
                       Confusion Matrix, ROC Curve, Feature Importance
  5. Model Persistence – saves best model + preprocessor for deployment

v4 Dataset Changes
------------------
  Term_Stage             : 1–4  (when in term the record was captured)
  CA1_Available          : 0/1 flag
  MidTerm_Available      : 0/1 flag
  Exam_Available         : 0/1 flag
  Current_Score_So_Far   : sum of completed assessment scores
  Max_Possible_So_Far    : max marks possible at this stage (0/20/40/100)
  Performance_Pct_So_Far : stage-normalised performance 0–1 (NaN at Stage 1)
  Subjects_Failed        : -1 = not yet determinable (Stages 1–3), 0–9 at Stage 4

  Raw scores (CA1_Score_Current, MidTerm_Score_Current, Exam_Score_Current)
  are EXCLUDED from model features; the engineered features carry that signal.

References
----------
  Federal Ministry of Information and National Orientation (FMINO). (2026,
    April 11). FG launches first-ever nationwide Learner Identification Number
    to transform education system. https://fmino.gov.ng/

  UNICEF. (2024). Nigeria – Out-of-school children: 18.3 million estimate.

  Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
    Journal of Machine Learning Research, 12, 2825–2830.

  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    KDD '16. https://doi.org/10.1145/2939672.2939785

  Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling
    Technique. JAIR, 16, 321–357.
================================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  NUTDTS 803 – EARLY WARNING ML MODEL FOR LEARNER DROPOUT  (v4)")
print("  Nigerian University of Technology and Management")
print("="*70)

DATA_PATH = "nigerian_student_dropout_dataset.csv"
df = pd.read_csv(DATA_PATH)

print(f"\n[1/5] DATA LOADING & EXPLORATION")
print(f"  Dataset shape  : {df.shape}")
print(f"  Columns        : {list(df.columns)}")
print(f"  Missing values : {df.isnull().sum().sum()}  (NaN in Performance_Pct_So_Far at Stage 1 is expected)")

vc = df["Dropout_Risk"].value_counts()
print(f"\n  Target distribution:")
print(f"    Not at Risk (0) : {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
print(f"    At Risk     (1) : {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")

# Stage distribution
stage_dist = df.groupby("Term_Stage").agg(
    Count=("Dropout_Risk", "count"),
    Dropout_Rate=("Dropout_Risk", "mean")
).reset_index()
stage_dist["Dropout_Rate_Pct"] = stage_dist["Dropout_Rate"] * 100
print("\n  Stage distribution:")
print(stage_dist.to_string(index=False))

# ── EDA Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis – Nigerian Student Dropout Dataset (v4)",
             fontsize=14, fontweight="bold", y=1.01)

# 1. Target distribution
axes[0,0].bar(["Not at Risk (0)", "At Risk (1)"],
              [vc[0], vc[1]], color=["#2ecc71","#e74c3c"], edgecolor="white")
axes[0,0].set_title("Target Variable Distribution")
axes[0,0].set_ylabel("Count")
for i, v in enumerate([vc[0], vc[1]]):
    axes[0,0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# 2. Dropout Rate by Term Stage
stage_labels = {1:"Stage 1\n(Pre-CA1)", 2:"Stage 2\n(Post-CA1)",
                3:"Stage 3\n(Post-MidTerm)", 4:"Stage 4\n(Post-Exam)"}
stage_drop = df.groupby("Term_Stage")["Dropout_Risk"].mean() * 100
axes[0,1].bar([stage_labels[s] for s in stage_drop.index],
              stage_drop.values, color=["#85c1e9","#3498db","#1a6fa8","#1a5276"],
              edgecolor="white")
axes[0,1].set_title("Dropout Rate by Term Stage (%)")
axes[0,1].set_ylabel("Dropout Rate (%)")
for i, v in enumerate(stage_drop.values):
    axes[0,1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)

# 3. Dropout by Income
inc_drop = df.groupby("Household_Income_Level")["Dropout_Risk"].mean() * 100
inc_order = ["Low","Medium","High"]
inc_drop = inc_drop.reindex(inc_order)
axes[0,2].bar(inc_order, inc_drop, color=["#e74c3c","#f39c12","#2ecc71"], edgecolor="white")
axes[0,2].set_title("Dropout Rate by Household Income (%)")
axes[0,2].set_ylabel("Dropout Rate (%)")

# 4. Performance_Pct_So_Far distribution by risk (Stages 2–4 only)
df_stage2plus = df[df["Term_Stage"] > 1]
df_stage2plus[df_stage2plus["Dropout_Risk"]==0]["Performance_Pct_So_Far"].plot(
    kind="hist", ax=axes[1,0], alpha=0.6, color="#2ecc71", label="Not at Risk", bins=25)
df_stage2plus[df_stage2plus["Dropout_Risk"]==1]["Performance_Pct_So_Far"].plot(
    kind="hist", ax=axes[1,0], alpha=0.6, color="#e74c3c", label="At Risk", bins=25)
axes[1,0].set_title("Performance_Pct_So_Far by Risk Group\n(Stages 2–4)")
axes[1,0].set_xlabel("Stage-Normalised Performance (0–1)")
axes[1,0].legend()

# 5. Attendance Rate vs Dropout
df.boxplot(column="Attendance_Rate", by="Dropout_Risk", ax=axes[1,1],
           patch_artist=True)
axes[1,1].set_title("Attendance Rate by Risk Group")
axes[1,1].set_xlabel("Dropout Risk (0=No, 1=Yes)")
axes[1,1].set_ylabel("Attendance Rate")
plt.sca(axes[1,1])
plt.title("Attendance Rate by Risk Group")

# 6. Correlation heatmap (numeric features used in model)
num_cols_eda = ["Term_Grade_Prev1","Term_Grade_Prev2","Term_Grade_Prev3",
                "Term_Stage","CA1_Available","MidTerm_Available","Exam_Available",
                "Current_Score_So_Far","Max_Possible_So_Far",
                "Attendance_Rate","Subjects_Failed","Distance_to_School_km",
                "Age","Dropout_Risk"]
corr_matrix = df[num_cols_eda].corr()
sns.heatmap(corr_matrix, ax=axes[1,2], cmap="RdYlGn", annot=False,
            linewidths=0.4, vmin=-1, vmax=1)
axes[1,2].set_title("Correlation Heatmap (Model Features)")

plt.tight_layout()
plt.savefig("outputs/01_eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ EDA plots saved → outputs/01_eda_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/5] PREPROCESSING")

# ── v4: Handle special values before feature selection ────────────────────────
# Performance_Pct_So_Far is NaN at Stage 1 (no assessments yet).
# Fill with -1 so the model learns "no data" as a distinct signal from 0.
df["Performance_Pct_So_Far"] = df["Performance_Pct_So_Far"].fillna(-1)

# Subjects_Failed = -1 at Stages 1–3 (not yet determinable) — keep as-is;
# the model will distinguish this from actual failed-subject counts.

# ── Drop columns not used in modelling ────────────────────────────────────────
# Identifiers / names
DROP_ALWAYS = ["Learner_Id_Number_LIN", "First_Name", "Last_Name"]

# Raw score columns: replaced by engineered stage features.
# Including them alongside the flags would create data-leakage artefacts
# (e.g. CA1_Score_Current = 0 at Stage 1 looks identical to a student who
#  truly scored 0 — the availability flag is what carries the distinction).
DROP_RAW_SCORES = ["CA1_Score_Current", "MidTerm_Score_Current", "Exam_Score_Current"]

DROP_COLS = DROP_ALWAYS + DROP_RAW_SCORES
df_model = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Separate features and target
X = df_model.drop(columns=["Dropout_Risk"])
y = df_model["Dropout_Risk"]

# Identify column types
CATEGORICAL_COLS = X.select_dtypes(include=["object"]).columns.tolist()
NUMERICAL_COLS   = X.select_dtypes(include=["number"]).columns.tolist()

print(f"  Features used  : {len(X.columns)}")
print(f"  Categorical    : {CATEGORICAL_COLS}")
print(f"  Numerical      : {NUMERICAL_COLS}")
print(f"  Excluded (raw scores + IDs): {DROP_COLS}")

# Build ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num",  StandardScaler(),                         NUMERICAL_COLS),
    ("cat",  OneHotEncoder(handle_unknown="ignore",
                           sparse_output=False),       CATEGORICAL_COLS),
], remainder="drop")

# Train / Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")

# Fit preprocessor on train, transform both
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# SMOTE – address class imbalance on training set only
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
print(f"  After SMOTE – Train: {X_train_bal.shape[0]:,} "
      f"(At Risk: {y_train_bal.sum():,} | Not: {(y_train_bal==0).sum():,})")

# Save preprocessor for deployment
joblib.dump(preprocessor, "outputs/preprocessor.pkl")
print("  ✓ Preprocessor saved → outputs/preprocessor.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] MODEL TRAINING")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced", C=1.0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum()
    ),
}

trained_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training: {name}")
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model

    cv_scores = cross_val_score(model, X_train_bal, y_train_bal,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"    5-Fold CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(model, f"outputs/model_{name.replace(' ','_').lower()}.pkl")
    print(f"    ✓ Saved → outputs/model_{name.replace(' ','_').lower()}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/5] EVALUATION")

results = {}
for name, model in trained_models.items():
    y_pred  = model.predict(X_test_proc)
    y_prob  = model.predict_proba(X_test_proc)[:, 1]
    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

results_df = pd.DataFrame(results).T[["Accuracy","Precision","Recall","F1","ROC-AUC"]]
print("\n  ── Model Comparison ──")
print(results_df.round(4).to_string())

# Identify best model by ROC-AUC
best_name  = results_df["ROC-AUC"].idxmax()
best_model = trained_models[best_name]
print(f"\n  ★ Best Model : {best_name}  (ROC-AUC = {results_df.loc[best_name,'ROC-AUC']:.4f})")
joblib.dump(best_model, "outputs/best_model.pkl")
print("  ✓ Best model saved → outputs/best_model.pkl")

# Save results table
results_df.round(4).to_csv("outputs/model_comparison.csv")

# ── Evaluation Plots ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("Model Evaluation – Nigerian Student Dropout Prediction (v4)",
             fontsize=14, fontweight="bold")

colors = {"Logistic Regression": "#3498db",
          "Random Forest":       "#2ecc71",
          "XGBoost":             "#e74c3c"}

# (a) Metric comparison bar chart
ax0 = fig.add_subplot(gs[0, :2])
metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
x = np.arange(len(metrics))
width = 0.25
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    ax0.bar(x + i*width, vals, width, label=name,
            color=colors[name], edgecolor="white", alpha=0.88)
ax0.set_xticks(x + width)
ax0.set_xticklabels(metrics)
ax0.set_ylim(0, 1.05)
ax0.set_ylabel("Score")
ax0.set_title("Model Performance Metrics Comparison")
ax0.legend()
ax0.axhline(0.80, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

# (b) ROC Curves
ax1 = fig.add_subplot(gs[0, 2])
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax1.plot(fpr, tpr, label=f"{name} ({res['ROC-AUC']:.3f})",
             color=colors[name], linewidth=2)
ax1.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.5)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves")
ax1.legend(fontsize=8)

# (c–e) Confusion Matrices
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not at Risk","At Risk"],
                yticklabels=["Not at Risk","At Risk"],
                linewidths=0.5)
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.savefig("outputs/02_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Evaluation plots saved → outputs/02_model_evaluation.png")

# ── Feature Importance (best model) ──────────────────────────────────────────
ohe_cats = preprocessor.named_transformers_["cat"]\
               .get_feature_names_out(CATEGORICAL_COLS).tolist()
feature_names = NUMERICAL_COLS + ohe_cats

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.zeros(len(feature_names))

fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
        color="#3498db", edgecolor="white")
ax.set_title(f"Top 15 Feature Importances – {best_name}", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/03_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Feature importance saved → outputs/03_feature_importance.png")

# ── Classification Reports ────────────────────────────────────────────────────
with open("outputs/classification_reports.txt", "w", encoding="utf-8") as f:
    f.write("NUTDTS 803 – ML Model Classification Reports (v4 Dataset)\n")
    f.write("Nigerian University of Technology and Management\n")
    f.write("="*60 + "\n\n")
    for name, res in results.items():
        f.write(f"{'─'*40}\n{name}\n{'─'*40}\n")
        f.write(classification_report(y_test, res["y_pred"],
                target_names=["Not at Risk","At Risk"]))
        f.write("\n")
print("  ✓ Classification reports → outputs/classification_reports.txt")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE METADATA FOR DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5/5] SAVING DEPLOYMENT METADATA")

metadata = {
    "best_model_name":   best_name,
    "categorical_cols":  CATEGORICAL_COLS,
    "numerical_cols":    NUMERICAL_COLS,
    "feature_names":     list(X.columns),
    "results_summary":   results_df.round(4).to_dict(),
    "dataset_version":   "v4",
    "dropped_raw_scores": DROP_RAW_SCORES,
}
joblib.dump(metadata, "outputs/metadata.pkl")

# Save column unique values for Streamlit dropdowns
col_options = {}
for col in CATEGORICAL_COLS:
    col_options[col] = sorted(df[col].dropna().unique().tolist())
joblib.dump(col_options, "outputs/col_options.pkl")

print("  ✓ Metadata saved      → outputs/metadata.pkl")
print("  ✓ Column options saved → outputs/col_options.pkl")

print("\n" + "="*70)
print("  PIPELINE COMPLETE  (v4 – stage-aware early warning)")
print(f"  Best Model : {best_name}")
print(f"  ROC-AUC    : {results_df.loc[best_name,'ROC-AUC']:.4f}")
print(f"  F1 Score   : {results_df.loc[best_name,'F1']:.4f}")
print("="*70 + "\n")
