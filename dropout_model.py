# Student Dropout Predictor
# By Padma Shree
# Project 10 of 25

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1 - Load data
print("=== STUDENT DROPOUT PREDICTOR ===")
df = pd.read_csv(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\student_data.csv")
# Step 2 - Understand data
print("Total records:", len(df))
print("Columns:", df.columns.tolist())
print("\nTarget distribution:")
print(df["Target"].value_counts())

# Step 3 - Prepare data for ML
print("\n=== TRAINING ML MODEL ===")

# Keep only Graduate and Dropout for binary classification
df_model = df[df["Target"] != "Enrolled"].copy()
df_model["Target"] = df_model["Target"].map({"Graduate": 0, "Dropout": 1})

# Features and target
X = df_model.drop("Target", axis=1)
y = df_model["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4 - Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5 - Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))

# Step 6 - Feature importance chart
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind="barh", color="purple", figsize=(10,6))
plt.title("Top 10 Factors That Predict Student Dropout")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\03_phase3\dropout_predictor\feature_importance.png")
plt.show()
print("\nChart saved!!")