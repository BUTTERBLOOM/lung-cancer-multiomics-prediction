# first_real_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('cancer_data.csv')
print("Data loaded!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:5]}...")

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"\nTarget distribution:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
top_features = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
print("\nTop 10 Most Important Features:")
print(top_features)

# Plot
plt.figure(figsize=(10,6))
plt.barh(range(10), top_features['importance'].values[::-1])
plt.yticks(range(10), top_features['feature'].values[::-1])
plt.xlabel('Importance')
plt.title('Top 10 Features for Breast Cancer Detection')
plt.tight_layout()
plt.savefig('cancer_top_features.png')
plt.show()

print("\n✅ Model complete! Check cancer_top_features.png")