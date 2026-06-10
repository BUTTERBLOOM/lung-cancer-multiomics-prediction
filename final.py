# lung_cancer_ml_final.py
# COMPLETE VERSION — includes external validation on GSE10072

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


print("Loading data...")
df = pd.read_csv(r"C:/Users/bhaav/OneDrive/Documents/ml_ready_data.csv")

print(f"Data shape: {df.shape}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Unique labels found: {y.unique()}")
y = y.map({'Normal': 0, 'Tumor': 1})

if y.isnull().any():
    raise ValueError("Label mapping failed. Check your condition column — expected 'Normal' and 'Tumor'.")

print(f"Features : {X.shape[1]} genes")
print(f"Samples  : {X.shape[0]} patients")
print(f"Classes  : {y.value_counts().to_dict()}")
print()


try:
    gene_map_df   = pd.read_csv(r"C:/Users/bhaav/OneDrive/Documents/gene_mapping.csv")
    probe_to_gene = dict(zip(gene_map_df['Probe_ID'], gene_map_df['Gene_Symbol']))
    print(f"Loaded {len(probe_to_gene)} gene mappings")
except FileNotFoundError:
    probe_to_gene = {}
    print("gene_mapping.csv not found — probe IDs will be used")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


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


results = [
    {'Model': 'Random Forest',       'Accuracy': rf_acc,  'AUC': rf_auc,  'F1': rf_f1},
    {'Model': 'SVM',                 'Accuracy': svm_acc, 'AUC': svm_auc, 'F1': svm_f1},
    {'Model': 'Logistic Regression', 'Accuracy': lr_acc,  'AUC': lr_auc,  'F1': lr_f1},
]

results_df = pd.DataFrame(results)
results_df[['Accuracy', 'AUC', 'F1']] = results_df[['Accuracy', 'AUC', 'F1']].round(4)

print("\n" + "=" * 50)
print("COMPLETE MODEL COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")


best_name  = 'Random Forest'
best_pred  = rf_pred
best_prob  = rf_prob


best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X, y)
print(f"\nUsing {best_name} for visualizations and external validation")


print("\n" + "=" * 50)
print(f"CLASSIFICATION REPORT — {best_name}")
print("=" * 50)
print(classification_report(y, best_pred, target_names=['Normal', 'Tumor']))


print("Extracting feature importance...")
importances = best_model.feature_importances_
top_idx    = np.argsort(importances)[-10:]
top_probes = X.columns[top_idx]

gene_labels = [probe_to_gene.get(probe, probe) for probe in top_probes]

print("\nTop 10 genes with real names:")
for i, (probe, gene, imp) in enumerate(
        zip(top_probes, gene_labels, importances[top_idx])):
    print(f"   {i+1:2d}. {gene:<12} ({probe})  importance={imp:.4f}")


print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Hybrid Lung Cancer Prediction — Results', fontsize=14, fontweight='bold')

ConfusionMatrixDisplay.from_predictions(
    y, best_pred,
    display_labels=['Normal', 'Tumor'],
    cmap='Blues',
    ax=axes[0, 0])
axes[0, 0].set_title(f'Confusion Matrix — {best_name}')

RocCurveDisplay.from_predictions(y, best_prob, ax=axes[0, 1])
axes[0, 1].set_title(f'ROC Curve — {best_name} (AUC = {roc_auc_score(y, best_prob):.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random Classifier')
axes[0, 1].legend(loc='lower right', fontsize=8)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')

colors = ['#e74c3c' if i == len(top_idx)-1 else '#3498db' for i in range(len(top_idx))]
axes[1, 0].barh(range(10), importances[top_idx], color=colors)
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels(gene_labels)
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title('Top 10 Predictive Genes (Random Forest)')
for i, imp in enumerate(importances[top_idx]):
    axes[1, 0].text(imp + 0.0005, i, f'{imp:.4f}', va='center', fontsize=8)

model_names = [r['Model'] for r in results]
acc_scores  = [r['Accuracy'] for r in results]
auc_scores  = [r['AUC']      for r in results]
f1_scores   = [r['F1']       for r in results]

x = np.arange(len(model_names))
w = 0.25
axes[1, 1].bar(x - w, acc_scores, w, label='Accuracy', color='steelblue')
axes[1, 1].bar(x,     auc_scores, w, label='AUC',      color='coral')
axes[1, 1].bar(x + w, f1_scores,  w, label='F1-Score', color='mediumseagreen')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(model_names, fontsize=9)
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Comparison (Accuracy / AUC / F1)')
axes[1, 1].legend(fontsize=8)
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores)):
    axes[1, 1].text(i - w, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i,     auc + 0.02, f'{auc:.3f}', ha='center', fontsize=8)
    axes[1, 1].text(i + w, f1  + 0.02, f'{f1:.3f}',  ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ml_results.png")


print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"Random Forest      : Accuracy={rf_acc:.4f}  AUC={rf_auc:.4f}  F1={rf_f1:.4f}")
print(f"SVM                : Accuracy={svm_acc:.4f}  AUC={svm_auc:.4f}  F1={svm_f1:.4f}")
print(f"Logistic Regression: Accuracy={lr_acc:.4f}  AUC={lr_auc:.4f}  F1={lr_f1:.4f}")
print(f"\nVisualization model: {best_name}")


print("\n" + "="*50)
print("DEMO: Single patient prediction")
print("="*50)

p1      = X.iloc[0:1]
pred1   = best_model.predict(p1)[0]
prob1   = best_model.predict_proba(p1)[0]
label1  = 'Normal' if pred1 == 0 else 'Tumor'
conf1   = prob1[pred1]
print(f"Patient 1  (Actual: Normal) → Predicted: {label1}  (Confidence: {conf1:.1%})")

p2      = X.iloc[60:61]
pred2   = best_model.predict(p2)[0]
prob2   = best_model.predict_proba(p2)[0]
label2  = 'Normal' if pred2 == 0 else 'Tumor'
conf2   = prob2[pred2]
print(f"Patient 61 (Actual: Tumor)  → Predicted: {label2}  (Confidence: {conf2:.1%})")


print("\n" + "="*50)
print("EXTERNAL VALIDATION ON GSE10072")
print("="*50)

import os

val_path = r"C:/Users/bhaav/OneDrive/Desktop/aiml/validation_data.csv"

if not os.path.exists(val_path):
    print("validation_data.csv not found.")
    print("Please run the R script for GSE10072 first to generate this file.")
    print("Expected location:", val_path)
else:
    # Load validation data
    val_df = pd.read_csv(val_path)
    print(f"Validation data shape : {val_df.shape}")
    print(f"Unique labels found   : {val_df.iloc[:, -1].unique()}")

    X_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]

  
    y_val_mapped = y_val.map({'Normal': 1, 'Tumor': 0})

    if y_val_mapped.isnull().any():
        print("Standard mapping failed. Trying alternate label names...")
        y_val_mapped = y_val.map({'normal': 1, 'tumor': 0,
                                   'Normal': 1, 'Tumor': 0,
                                   'non-tumor': 1, 'tumor': 0,
                                   'Non-tumor': 1, 'Tumor': 0,
                                   'adjacent normal': 1, 'lung tumor': 0,
                                   'Adjacent Normal': 1, 'Lung Tumor': 0})

    if y_val_mapped.isnull().any():
        print("\nCould not map labels automatically.")
        print("Actual label values:", y_val.unique())
        print("Please paste those label names here and I will fix the mapping.")
    else:
        y_val = y_val_mapped
        print(f"Labels mapped: {y_val.value_counts().to_dict()}")

        train_cols   = list(X.columns)
        common_cols  = [c for c in train_cols if c in X_val.columns]
        missing      = len(train_cols) - len(common_cols)

        print(f"\nTraining features    : {len(train_cols)}")
        print(f"Common features      : {len(common_cols)}")
        print(f"Missing in validation: {missing}")

        if len(common_cols) < 40:
            print("WARNING: fewer than 40 common genes found.")
            print("GSE10072 may use a different platform. Check your R export.")
        else:
            X_val_aligned = X_val[common_cols]
            X_train_aligned = X[common_cols]

            val_model = RandomForestClassifier(n_estimators=100, random_state=42)
            val_model.fit(X_train_aligned, y)

            y_pred_val = val_model.predict(X_val_aligned)
            y_prob_val = val_model.predict_proba(X_val_aligned)[:, 1]

            val_acc = accuracy_score(y_val, y_pred_val)
            val_auc = roc_auc_score(y_val, y_prob_val)
            val_f1  = f1_score(y_val, y_pred_val)

            print("\n" + "="*50)
            print("EXTERNAL VALIDATION RESULTS (GSE10072)")
            print("="*50)
            print(f"Accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
            print(f"AUC      : {val_auc:.4f}")
            print(f"F1-Score : {val_f1:.4f}")
            print()
            print(classification_report(y_val, y_pred_val,
                  target_names=['Normal', 'Tumor']))

            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle('External Validation — GSE10072 (Independent Dataset)',
                          fontsize=13, fontweight='bold')

            ConfusionMatrixDisplay.from_predictions(
                y_val, y_pred_val,
                display_labels=['Normal', 'Tumor'],
                cmap='Greens',
                ax=axes2[0])
            axes2[0].set_title('Confusion Matrix — GSE10072')

            RocCurveDisplay.from_predictions(y_val, y_prob_val, ax=axes2[1])
            axes2[1].set_title(f'ROC Curve — GSE10072 (AUC = {val_auc:.3f})')
            axes2[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8)
            axes2[1].set_xlabel('False Positive Rate')
            axes2[1].set_ylabel('True Positive Rate')

            plt.tight_layout()
            plt.savefig('external_validation.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: external_validation.png")

            print("\n" + "="*50)
            print("COMPARISON: Training vs External Validation") 
            print("="*50)
            print(f"{'Metric':<12} {'GSE19804 (train)':>18} {'GSE10072 (external)':>20}")
            print("-"*52)
            print(f"{'Accuracy':<12} {rf_acc*100:>17.2f}% {val_acc*100:>19.2f}%")
            print(f"{'AUC':<12} {rf_auc:>18.4f} {val_auc:>20.4f}")
            print(f"{'F1-Score':<12} {rf_f1:>18.4f} {val_f1:>20.4f}")
            print()

            drop = (rf_acc - val_acc) * 100
            if drop < 10:
                print(f"Accuracy drop of {drop:.1f}% on external data — GOOD generalisation.")
            elif drop < 20:
                print(f"Accuracy drop of {drop:.1f}% on external data — acceptable generalisation.")
            else:
                print(f"Accuracy drop of {drop:.1f}% — consider retraining with more data.")

print("\n" + "="*50)
print("ALL DONE")
print("="*50)
print("Files saved:")
print("   results.png          — main results (4 plots)")
print("   model_comparison.csv    — all 3 model scores")
if os.path.exists(val_path):
    print("   external_validation.png — GSE10072 validation results")