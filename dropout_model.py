# Student Dropout Predictor
# By Padma Shree
# Project 10 of 25

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("🎓 STUDENT DROPOUT PREDICTOR")
print("=" * 50)

# Step 1: Load data
file_path = r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\student_data.csv"

try:
    df = pd.read_csv(file_path)
    print(f"\n✅ Dataset loaded: {len(df)} students")
except FileNotFoundError:
    print("\n❌ File not found!")
    exit()

# Step 2: Target column is 'Target' (Dropout, Graduate, Enrolled)
target_col = 'Target'
print(f"\n🎯 Target column: {target_col}")
print(f"Unique values: {df[target_col].unique()}")

# Step 3: Filter only Dropout and Graduate (remove Enrolled)
df = df[df[target_col].isin(['Dropout', 'Graduate'])]
print(f"\n📊 After filtering (Dropout + Graduate only): {len(df)} students")

# Step 4: Encode target (Dropout=1, Graduate=0)
df[target_col] = df[target_col].map({'Graduate': 0, 'Dropout': 1})

# Step 5: Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Step 6: Handle categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Step 7: Handle missing values
X = X.fillna(X.median())

print(f"\n📊 Features after preprocessing: {X.shape[1]}")
print(f"🎯 Students who dropped out: {(y == 1).sum()}")
print(f"✅ Students who graduated: {(y == 0).sum()}")

# Step 8: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📚 Training data: {len(X_train)} students")
print(f"🧪 Testing data: {len(X_test)} students")

# Step 9: Train Decision Tree model
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("\n🤖 Decision Tree model trained!")

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"True Graduated: {cm[0,0]} | False Dropout: {cm[0,1]}")
print(f"False Graduated: {cm[1,0]} | True Dropout: {cm[1,1]}")

# ============================================
# CHART 1: Confusion Matrix Heatmap
# ============================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Graduated', 'Dropout'],
            yticklabels=['Graduated', 'Dropout'])
plt.title('Confusion Matrix - Student Dropout Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Chart 1 saved: confusion_matrix.png")

# ============================================
# CHART 2: Feature Importance
# ============================================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Top 10 Factors Affecting Student Dropout')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Chart 2 saved: feature_importance.png")

# ============================================
# CHART 3: Outcome Distribution
# ============================================
outcome_counts = y.value_counts()
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Graduated', 'Dropped Out'], 
        autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Student Outcomes: Graduated vs Dropped Out')
plt.tight_layout()
plt.savefig('outcome_distribution.png')
plt.show()
print("✅ Chart 3 saved: outcome_distribution.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 50)
print("📋 SUMMARY OF FINDINGS")
print("=" * 50)
print(f"✅ Total Students Analyzed: {len(df)}")
print(f"✅ Students Who Graduated: {(y == 0).sum()}")
print(f"✅ Students Who Dropped Out: {(y == 1).sum()}")
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"\n🔍 Top 3 Factors Influencing Dropout:")
for i in range(min(3, len(feature_importance))):
    print(f"   {i+1}. {feature_importance.iloc[i]['Feature']}")

print("\n" + "=" * 50)
print("✅ PROJECT 10 COMPLETE! 3 charts saved!")
print("=" * 50)