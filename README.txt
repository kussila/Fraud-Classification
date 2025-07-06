# 💳 Credit Card Fraud Detection with Random Forest

## 📚 Project Description

This project detects fraudulent credit card transactions using a supervised machine learning approach. The dataset is highly imbalanced, making it a real-world anomaly detection problem. We trained a **Random Forest Classifier** optimized for performance and robustness.

---

## 📁 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Transactions made by European cardholders in September 2013.
- **Total Samples**: 284,807 transactions  
- **Fraudulent Cases**: 492 (~0.172%)  
- **Features**:
  - `Time`: Time elapsed from the first transaction  
  - `Amount`: Transaction amount  
  - `V1` to `V28`: Principal components (PCA-transformed)  
  - `Class`: Target (0 = Legitimate, 1 = Fraudulent)

---

## 🧠 Model Used

### `RandomForestClassifier` Configuration:

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

---

## 📊 Model Evaluation Metrics

| Metric                           | Value     |
|----------------------------------|-----------|
| **Accuracy**                     | 0.9996    |
| **Precision**                    | 0.9747    |
| **Recall (Sensitivity)**         | 0.7857    |
| **F1 Score**                     | 0.8701    |
| **Matthews Correlation Coefficient (MCC)** | 0.8749 |

📌 **Interpretation**:
- **High accuracy** is expected due to class imbalance.
- **Precision** is high: most predicted frauds are true frauds.
- **Recall** is moderate: some frauds are missed.
- **F1 score** balances precision and recall.
- **MCC** gives a reliable measure even with class imbalance.

---

## ⏱️ Performance Timing

| Phase              | Time (seconds) |
|--------------------|----------------|
| Training           | 375.41         |
| Prediction         | 0.94           |

---

## 📦 Exported Artifacts

- `random_forest_model_fraud_classification.pkl`: Trained Random Forest model
- `features.json`: Feature list used during training

---

## 🚀 Usage Guide

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn joblib
```

---

### 2️⃣ Load Model and Features

```python
import joblib
import json
import pandas as pd

# Load the trained model
model = joblib.load("random_forest_model_fraud_classification.pkl")

# Load the feature list
with open("features.json", "r") as f:
    features = json.load(f)
```

---

### 3️⃣ Prepare Input Data

```python
# Load your new transaction data
df = pd.read_csv("your_new_transactions.csv")

# Filter to keep only relevant features
df = df[features]
```

---

### 4️⃣ Make Predictions

```python
# Predict classes
predictions = model.predict(df)

# Predict fraud probability
probabilities = model.predict_proba(df)[:, 1]

print(predictions)
print(probabilities)
```

---

## 📌 Notes

- Due to the **high class imbalance**, precision and recall should always be monitored.
- Adjust the decision threshold to optimize for recall or precision depending on your business needs.
- The model generalizes well but should be retrained periodically with new data.

---

## 🙏 Acknowledgements

- Dataset provided by ULB & Worldline  
- Original research: *Dal Pozzolo et al.*  
- [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📃 License

MIT License – free to use, modify, and distribute with attribution.