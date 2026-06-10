import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load validation data
val_df = pd.read_csv("C:\\Users\\bhaav\\OneDrive\\Desktop\\aiml\\external_ml_data.csv")
X_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1].map({'Normal': 0, 'Tumor': 1})

# Your already trained model predicts on completely new patients
y_pred_val = best_model.predict(X_val)
y_prob_val = best_model.predict_proba(X_val)[:, 1]

print("EXTERNAL VALIDATION RESULTS")
print(f"Accuracy : {accuracy_score(y_val, y_pred_val):.4f}")
print(f"AUC      : {roc_auc_score(y_val, y_prob_val):.4f}")
print(classification_report(y_val, y_pred_val, 
      target_names=['Normal', 'Tumor']))