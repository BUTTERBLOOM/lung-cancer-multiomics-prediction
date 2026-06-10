# Each model gets its own independent predictions
# FIXED VERSION: ROC title matches model, gene names shown

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
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
# CROSS-VALIDATION SETUP
# ============================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================
# HELPER: run one model, print + return metrics
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
# RUN ALL THREE MODELS INDEPENDENTLY
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
# RESULTS TABLE (now includes F1)
# ============================================
results = [
    {'Model': 'Random Forest',      'Accuracy': rf_acc,  'AUC': rf_auc,  'F1': rf_f1},
    {'Model': 'SVM',                'Accuracy': svm_acc, 'AUC': svm_auc, 'F1': svm_f1},
    {'Model': 'Logistic Regression','Accuracy': lr_acc,  'AUC': lr_auc,  'F1': lr_f1},
]

results_df = pd.DataFrame(results)
# Round for clean display
results_df[['Accuracy','AUC','F1']] = results_df[['Accuracy','AUC','F1']].round(4)

print("\n" + "=" * 50)
print("COMPLETE MODEL COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

# ============================================
# PICK BEST MODEL BY AUC (more reliable than accuracy)
# ============================================
best_idx  = np.argmax([rf_auc, svm_auc, lr_auc])
all_preds = [rf_pred,  svm_pred,  lr_pred]
all_probs = [rf_prob,  svm_prob,  lr_prob]
all_names = ['Random Forest', 'SVM', 'Logistic Regression']

best_name = all_names[best_idx]
best_pred = all_preds[best_idx]
best_prob = all_probs[best_idx]
print(f"\nBest model (by AUC): {best_name}")

# ============================================
# CLASSIFICATION REPORT — BEST MODEL
# ============================================
print("\n" + "=" * 50)
print(f"CLASSIFICATION REPORT — {best_name}")
print("=" * 50)
print(classification_report(y, best_pred, target_names=['Normal', 'Tumor']))

# ============================================
# FEATURE IMPORTANCE (Random Forest only)
# Train on full data ONLY for importance — NOT used for evaluation metrics
# ============================================
print("Fitting Random Forest on full data for feature importance...")
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X, y)
importances = rf_full.feature_importances_
top_idx     = np.argsort(importances)[-10:]
top_probes  = X.columns[top_idx]

# FIX 2: Load gene names properly
try:
    gmap = pd.read_csv("gene_mapping.csv")
    # Create mapping dictionary
    p2g = dict(zip(gmap['Probe_ID'], gmap['Gene_Symbol']))
    # Get gene names for top probes, fallback to probe ID if not found
    gene_labels = []
    for p in top_probes:
        gene_name = p2g.get(p, p)
        # Clean up any .1, .2 suffixes
        gene_name = gene_name.split('.')[0]
        gene_labels.append(gene_name[:20])
    print("✅ Loaded real gene names from gene_mapping.csv")
    print(f"   Top gene: {gene_labels[0]} (probe: {top_probes[0]})")
except Exception as e:
    gene_labels = [p[:15] for p in top_probes]
    print(f"⚠️ Using probe IDs (gene_mapping.csv not found): {e}")

# ============================================
# VISUALIZATIONS (2 × 2 grid)
# ============================================
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Hybrid Lung Cancer Prediction — Results', fontsize=14, fontweight='bold')

# --- Plot 1: Confusion Matrix (FIXED: shows correct best model name) ---
ConfusionMatrixDisplay.from_predictions(
    y, best_pred,
    display_labels=['Normal', 'Tumor'],
    cmap='Blues',
    ax=axes[0, 0])
axes[0, 0].set_title(f'Confusion Matrix — {best_name}')

# --- Plot 2: ROC Curve (FIXED: now shows correct model name, not hardcoded) ---
RocCurveDisplay.from_predictions(y, best_prob, ax=axes[0, 1])
axes[0, 1].set_title(f'ROC Curve — {best_name} (AUC = {roc_auc_score(y, best_prob):.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random Classifier')
axes[0, 1].legend(loc='lower right', fontsize=8)

# --- Plot 3: Feature Importance (FIXED: shows real gene names) ---
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(10)]  # highlight top gene
axes[1, 0].barh(range(10), importances[top_idx], color=colors)
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels(gene_labels)
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title('Top 10 Predictive Genes (Random Forest)')

# Add importance values on bars
for i, (imp, label) in enumerate(zip(importances[top_idx], gene_labels)):
    axes[1, 0].text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=7)

# --- Plot 4: Model Comparison — shows Accuracy, AUC AND F1 ---
model_names = [r['Model'] for r in results]
acc_scores  = [r['Accuracy'] for r in results]
auc_scores  = [r['AUC']      for r in results]
f1_scores   = [r['F1']       for r in results]

x = np.arange(len(model_names))
w = 0.25  # bar width
axes[1, 1].bar(x - w,   acc_scores, w, label='Accuracy', color='steelblue')
axes[1, 1].bar(x,       auc_scores, w, label='AUC',      color='coral')
axes[1, 1].bar(x + w,   f1_scores,  w, label='F1-Score', color='mediumseagreen')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(model_names, fontsize=9)
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Comparison (Accuracy / AUC / F1)')
axes[1, 1].legend(fontsize=8)

# Value labels on every bar
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores)):
    axes[1, 1].text(i - w, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i,     auc + 0.02, f'{auc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i + w, f1  + 0.02, f'{f1:.3f}',  ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"Random Forest      : Accuracy={rf_acc:.4f}  AUC={rf_auc:.4f}  F1={rf_f1:.4f}")
print(f"SVM                : Accuracy={svm_acc:.4f}  AUC={svm_auc:.4f}  F1={svm_f1:.4f}")
print(f"Logistic Regression: Accuracy={lr_acc:.4f}  AUC={lr_auc:.4f}  F1={lr_f1:.4f}")
print(f"\nBest model (AUC)   : {best_name}")
print("\n✅ Files saved:")
print("   - ml_results.png (all plots)")
print("   - model_comparison.csv (performance table)")