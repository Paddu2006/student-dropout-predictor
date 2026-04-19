# Student Dropout Predictor
# By Padma Shree
# Project 11 of 25

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Step 1 - Load data
print("=== STUDENT DROPOUT PREDICTOR ===")
df = pd.read_csv(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\student_dropout.csv")

# Step 2 - Understand data
print("Total records:", len(df))
print("Columns:", df.columns.tolist())
print("\nTarget distribution:")
print(df["Target"].value_counts())

# Step 3 - Prepare data
print("\n=== TRAINING ML MODEL ===")
df_model = df[df["Target"] != "Enrolled"].copy()
df_model["Target"] = df_model["Target"].map({"Graduate": 0, "Dropout": 1})

X = df_model.drop("Target", axis=1)
y = df_model["Target"]

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

# Step 6 - Charts
# Chart 1 - Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind="barh", color="purple", figsize=(10,6))
plt.title("Top 10 Factors That Predict Student Dropout")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\03_phase3\dropout_predictor\feature_importance.png")
plt.show()
print("Chart 1 saved!!")

# Chart 2 - Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"])
plt.title("Confusion Matrix - Student Dropout Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\03_phase3\dropout_predictor\confusion_matrix.png")
plt.show()
print("Chart 2 saved!!")

# Chart 3 - Outcome distribution
df["Target"].value_counts().plot(kind="pie", autopct="%1.1f%%",
                                  colors=["green", "red", "orange"],
                                  figsize=(8,8))
plt.title("Student Outcome Distribution")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\03_phase3\dropout_predictor\outcome_distribution.png")
plt.show()
print("Chart 3 saved!!")