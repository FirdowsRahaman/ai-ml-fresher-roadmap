# Module 08: Practical Scenarios & Fresher Coding Challenges

This module covers real-world problem-solving questions and simple coding exercises that freshers are commonly given during technical interviews and take-home assignments.

---

## 🛠️ Section 1: Common Interview Scenarios & How to Answer Them

### Scenario 1: "Your model has 99% accuracy. Is it a good model?"
**Best Fresher Answer:**
**Not necessarily!** Accuracy alone is misleading, especially with **imbalanced datasets**.

**Example:** 
- You build a model to detect cancer. 
- Only 1% of patients in your dataset actually have cancer.
- If the model just predicts "No Cancer" for **every patient**, it achieves 99% accuracy but misses every single cancer case — completely useless and dangerous!

**What to check instead:**
```python
from sklearn.metrics import classification_report, confusion_matrix

# Never rely on accuracy alone for imbalanced datasets
print(classification_report(y_true, y_pred))
# This shows: Precision, Recall, F1-Score per class
```

**Better metrics for imbalanced data:** F1-Score, Precision, Recall, AUC-ROC.

---

### Scenario 2: "Your model performs well on training data but poorly on test data. What do you do?"
**Best Fresher Answer:**
This is called **Overfitting**. The model memorized the training data (including its noise) instead of learning the real underlying patterns.

**Steps I would take:**
1. **Visualize the learning curves** (training loss vs. validation loss over epochs) to confirm overfitting.
2. **Add more training data** — overfitting happens because the model has too little data relative to its complexity.
3. **Apply Dropout** — randomly deactivate neurons during training (for neural networks).
4. **Reduce model complexity** — fewer layers/neurons (neural nets) or reduce max_depth (decision trees).
5. **Apply regularization** — L1 or L2 penalty to prevent large weights.
6. **Use Early Stopping** — stop training when validation performance stops improving.

---

### Scenario 3: "How would you approach a new ML problem from scratch?"
**Best Fresher Answer (Structured Response):**
I would follow this systematic approach:

```
Step 1: Understand the problem
  → Is it classification, regression, or clustering?
  → What metric matters? (Accuracy? Recall? Revenue impact?)

Step 2: Explore the data (EDA)
  → How many rows, columns, data types?
  → Missing values? Outliers? Class imbalance?

Step 3: Clean and preprocess the data
  → Handle missing values (impute or drop)
  → Encode categorical variables
  → Scale/normalize numerical features

Step 4: Start simple
  → Try Logistic Regression or Decision Tree first
  → Establish a baseline metric

Step 5: Iterate and improve
  → Try more powerful models (Random Forest, XGBoost)
  → Feature engineering — create new useful features
  → Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)

Step 6: Evaluate properly
  → Use cross-validation, not just a single train-test split
  → Report Precision, Recall, F1 (not just accuracy for imbalanced data)

Step 7: Deploy and monitor
  → Serve as an API, containerize with Docker, track performance over time
```

---

### Scenario 4: "What would you do if you receive data with 40% missing values in one column?"
**Best Fresher Answer:**
I would first analyze **why** the data is missing (random accident, or systematic):

1. **If it's a numerical column:**
   - If not many outliers → Fill with **mean**.
   - If outliers present → Fill with **median** (more robust).
   - If 40% is too much → Drop the column, or create a new feature `column_is_missing` (1/0) to signal missingness.
2. **If it's a categorical column:**
   - Fill with the **mode** (most frequent value), OR
   - Create a new category `"Unknown"` to preserve information.
3. **Never fill the target/label column** — if a row is missing the label, drop that row.

```python
import pandas as pd

df = pd.DataFrame({'age': [25, None, 30, None, 28], 'salary': [50000, 60000, None, 55000, 65000]})

# Check missing percentages
print(df.isnull().mean() * 100)  # Shows % missing per column

# Impute numerical with median
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)
```

---

## 💻 Section 2: Fresher Coding Challenges

### Challenge 1: Implement a Simple Linear Regression from Scratch
**Prompt:** Write a class that computes predictions using the linear regression formula without using sklearn.

```python
import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weight = 0.0
        self.bias = 0.0

    def predict(self, X):
        return self.weight * X + self.bias

    def fit(self, X, y):
        n = len(X)
        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            error = y_pred - y

            # Gradient descent update
            dw = (2 / n) * np.dot(error, X)
            db = (2 / n) * np.sum(error)

            self.weight -= self.lr * dw
            self.bias   -= self.lr * db

        return self

# Test it
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)  # y = 2x

model = SimpleLinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

print(f"Weight (slope): {model.weight:.2f}")  # Should be ~2.0
print(f"Bias:           {model.bias:.2f}")    # Should be ~0.0
print(f"Predict(6):     {model.predict(6):.2f}") # Should be ~12.0
```

---

### Challenge 2: Compute Precision, Recall, and F1 from Scratch
**Prompt:** Given lists of actual and predicted labels, compute the key evaluation metrics.

```python
def classification_metrics(y_true, y_pred):
    """
    Compute Accuracy, Precision, Recall, and F1-Score
    for binary classification (0 or 1 labels).
    """
    tp = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1)
    tn = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0)

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Confusion Matrix:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy:          {accuracy:.2%}")
    print(f"Precision:         {precision:.2%}")
    print(f"Recall:            {recall:.2%}")
    print(f"F1-Score:          {f1:.2%}")

# Test
y_true = [1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

classification_metrics(y_true, y_pred)
# Expected Output:
#   TP=3, TN=3, FP=1, FN=1
#   Accuracy:  75.00%
#   Precision: 75.00%
#   Recall:    75.00%
#   F1-Score:  75.00%
```

---

### Challenge 3: Clean a Messy Dataset (Pandas Task)
**Prompt:** Given a DataFrame with issues, clean it — handle missing values, remove duplicates, and fix data types.

```python
import pandas as pd
import numpy as np

# Messy dataset
df = pd.DataFrame({
    'name':    ['Alice', 'Bob', 'Alice', 'Charlie', None, 'Dave'],
    'age':     [25, None, 25, 30, 28, -1],   # -1 is invalid age
    'salary':  ['50000', '60000', '50000', '75000', '55000', 'N/A'],
    'joined':  ['2022-01-01', '2022-03-15', '2022-01-01', '2023-07-20', '2021-11-01', '2023-09-10']
})

print("Before cleaning:")
print(df)
print()

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Drop rows where 'name' is missing (can't have anonymous employees)
df = df.dropna(subset=['name'])

# 3. Fill missing age with median
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Convert to numeric first
df['age'] = df['age'].replace(-1, np.nan)               # -1 is invalid → NaN
df['age'].fillna(df['age'].median(), inplace=True)

# 4. Clean salary column — remove 'N/A', convert to numeric
df['salary'] = df['salary'].replace('N/A', np.nan)
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
df['salary'].fillna(df['salary'].median(), inplace=True)

# 5. Convert 'joined' to datetime
df['joined'] = pd.to_datetime(df['joined'])

print("After cleaning:")
print(df)
print(df.dtypes)
```

---

### Challenge 4: Use a Pre-trained LLM API (OpenAI / Gemini Style)
**Prompt:** Write a simple function that sends a question to an LLM API and returns the answer.

```python
# Using OpenAI's API (applicable to any LLM API with minor changes)
from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

def ask_llm(question: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Send a question to an LLM and return the response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.7,   # Creativity level (0 = deterministic, 1 = creative)
        max_tokens=500     # Maximum length of response
    )
    return response.choices[0].message.content

# Example usage
answer = ask_llm("Explain overfitting in machine learning in one paragraph.")
print(answer)

# With few-shot prompting
sentiment_prompt = """
You are a sentiment classifier. Classify the review as POSITIVE or NEGATIVE only.
Examples:
  Review: 'Great product!' → POSITIVE
  Review: 'Broke after a day.' → NEGATIVE
"""

result = ask_llm("The battery life is terrible.", system_prompt=sentiment_prompt)
print(result)  # NEGATIVE
```
