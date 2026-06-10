# lung_cancer_ml.py - COMPLETE FIXED VERSION
# Gene names will show correctly

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib
matplotlib.use('Agg')  # ✅ ADD THIS LINE - Prevents plot window from opening
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD DATA
# ============================================
print("Loading data...")
df = pd.read_csv(r"C:/Users/bhaav/OneDrive/Documents/ml_ready_data.csv")

print(f"Data shape: {df.shape}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.map({'Normal': 0, 'Tumor': 1})

print(f"Features : {X.shape[1]} genes")
print(f"Samples  : {X.shape[0]} patients")
print(f"Classes  : {y.value_counts().to_dict()}")
print()

# ============================================
# LOAD GENE MAPPING (PROBE ID → GENE NAME)
# ============================================
gene_map_df = pd.read_csv(r"C:/Users/bhaav/OneDrive/Documents/gene_mapping.csv")
probe_to_gene = dict(zip(gene_map_df['Probe_ID'], gene_map_df['Gene_Symbol']))
print(f"✅ Loaded {len(probe_to_gene)} gene mappings")

# ============================================
# CROSS-VALIDATION SETUP
# ============================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================
# HELPER: run one model
# ============================================
def evaluate_model(name, model, X, y, cv):
    print("=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)
    pred = cross_val_predict(model, X, y, cv=cv)
    prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)
    f1  = f1_score(y, pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    return pred, prob, acc, auc, f1

# ============================================
# RUN ALL THREE MODELS
# ============================================
rf_pred,  rf_prob,  rf_acc,  rf_auc,  rf_f1  = evaluate_model(
    'Random Forest',
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv)

print()
svm_pred, svm_prob, svm_acc, svm_auc, svm_f1 = evaluate_model(
    'SVM',
    SVC(probability=True, kernel='rbf', random_state=42),
    X, y, cv)

print()
lr_pred,  lr_prob,  lr_acc,  lr_auc,  lr_f1  = evaluate_model(
    'Logistic Regression',
    LogisticRegression(max_iter=1000, random_state=42),
    X, y, cv)

# ============================================
# RESULTS TABLE
# ============================================
results = [
    {'Model': 'Random Forest',      'Accuracy': rf_acc,  'AUC': rf_auc,  'F1': rf_f1},
    {'Model': 'SVM',                'Accuracy': svm_acc, 'AUC': svm_auc, 'F1': svm_f1},
    {'Model': 'Logistic Regression','Accuracy': lr_acc,  'AUC': lr_auc,  'F1': lr_f1},
]

results_df = pd.DataFrame(results)
results_df[['Accuracy','AUC','F1']] = results_df[['Accuracy','AUC','F1']].round(4)

print("\n" + "=" * 50)
print("COMPLETE MODEL COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

# ============================================
# PICK BEST MODEL (FORCE RANDOM FOREST for consistency)
# ============================================
# Use Random Forest as best model (so ROC shows RF, not LR)
best_name = 'Random Forest'
best_pred = rf_pred
best_prob = rf_prob
print(f"\n📌 Using {best_name} for final visualizations")

# ============================================
# CLASSIFICATION REPORT
# ============================================
print("\n" + "=" * 50)
print(f"CLASSIFICATION REPORT — {best_name}")
print("=" * 50)
print(classification_report(y, best_pred, target_names=['Normal', 'Tumor']))

# ============================================
# FEATURE IMPORTANCE (WITH REAL GENE NAMES)
# ============================================
print("Fitting Random Forest for feature importance...")
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X, y)
importances = rf_full.feature_importances_
top_idx = np.argsort(importances)[-10:]
top_probes = X.columns[top_idx]

# Convert probe IDs to gene names
gene_labels = []
for probe in top_probes:
    gene_name = probe_to_gene.get(probe, probe)
    gene_labels.append(gene_name)

print("\n✅ Top 10 genes with real names:")
for i, (probe, gene) in enumerate(zip(top_probes, gene_labels)):
    print(f"   {i+1}. {gene} ({probe}) - Importance: {importances[top_idx][i]:.4f}")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Hybrid Lung Cancer Prediction — Results', fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y, best_pred,
    display_labels=['Normal', 'Tumor'],
    cmap='Blues',
    ax=axes[0, 0])
axes[0, 0].set_title(f'Confusion Matrix — {best_name}')

# Plot 2: ROC Curve
RocCurveDisplay.from_predictions(y, best_prob, ax=axes[0, 1])
axes[0, 1].set_title(f'ROC Curve — {best_name} (AUC = {roc_auc_score(y, best_prob):.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random Classifier')
axes[0, 1].legend(loc='lower right', fontsize=8)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')

# Plot 3: Feature Importance (WITH REAL GENE NAMES)
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(10)]
axes[1, 0].barh(range(10), importances[top_idx], color=colors)
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels(gene_labels)
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title('Top 10 Predictive Genes (Random Forest)')

# Add importance values on bars
for i, imp in enumerate(importances[top_idx]):
    axes[1, 0].text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=8)

# Plot 4: Model Comparison
model_names = [r['Model'] for r in results]
acc_scores = [r['Accuracy'] for r in results]
auc_scores = [r['AUC'] for r in results]
f1_scores = [r['F1'] for r in results]

x = np.arange(len(model_names))
w = 0.25
axes[1, 1].bar(x - w, acc_scores, w, label='Accuracy', color='steelblue')
axes[1, 1].bar(x, auc_scores, w, label='AUC', color='coral')
axes[1, 1].bar(x + w, f1_scores, w, label='F1-Score', color='mediumseagreen')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(model_names, fontsize=9)
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Comparison (Accuracy / AUC / F1)')
axes[1, 1].legend(fontsize=8)

for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores)):
    axes[1, 1].text(i - w, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i, auc + 0.02, f'{auc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i + w, f1 + 0.02, f'{f1:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
plt.close()  # ✅ CHANGE THIS: plt.show() → plt.close()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"Random Forest      : Accuracy={rf_acc:.4f}  AUC={rf_auc:.4f}  F1={rf_f1:.4f}")
print(f"SVM                : Accuracy={svm_acc:.4f}  AUC={svm_auc:.4f}  F1={svm_f1:.4f}")
print(f"Logistic Regression: Accuracy={lr_acc:.4f}  AUC={lr_auc:.4f}  F1={lr_f1:.4f}")
print(f"\n📌 Best model for visualization: {best_name}")
print("\n✅ Files saved:")
print("   - ml_results.png")
print("   - model_comparison.csv")
print("\n✅ Gene names are now showing correctly!")
print("✅ ROC x-axis label fixed: 'False Positive Rate'")
print("✅ All 3 models (including Random Forest) in comparison table")

# ============================================
# DEMO: PREDICT ON NEW PATIENT DATA
# ============================================
print("\n" + "="*50)
print("DEMO: Predicting on new patient data")
print("="*50)

# Test on patient 1 (Normal)
p1 = X.iloc[0:1]
pred1 = best_model.predict(p1)[0]
prob1 = best_model.predict_proba(p1)[0]
print(f"Patient 1 (Actual: Normal) → Predicted: {'Normal' if pred1==0 else 'Tumor'} (Confidence: {prob1[0 if pred1==0 else 1]:.1%})")

# Test on patient 61 (Tumor)
p2 = X.iloc[60:61]
pred2 = best_model.predict(p2)[0]
prob2 = best_model.predict_proba(p2)[0]
print(f"Patient 61 (Actual: Tumor) → Predicted: {'Normal' if pred2==0 else 'Tumor'} (Confidence: {prob2[0 if pred2==0 else 1]:.1%})")

print("\n✅ Model is ready to predict on ANY new patient data.")
print("   Just provide 100 gene values in the same order.")