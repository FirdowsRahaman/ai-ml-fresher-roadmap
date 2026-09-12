# Module 03: Classical Machine Learning for Freshers

This module covers the core machine learning concepts and algorithms asked in every fresher AI/ML interview.

---

## Section 1: What is Machine Learning?

### Q1: What is Artificial Intelligence (AI)?
**Answer:**
Artificial Intelligence (AI) is the ability of a computer or machine to perform tasks that would normally require **human intelligence** — like recognizing faces, understanding speech, translating languages, or making decisions.

**Examples of AI in daily life:**
- Google Maps predicting traffic and suggesting routes
- Netflix recommending shows you might like
- Spam detection in your Gmail inbox
- Face unlock on your phone

---

### Q2: What is Machine Learning (ML)? How is it different from traditional programming?
**Answer:**

| Traditional Programming | Machine Learning |
|---|---|
| Developer writes explicit rules (IF-THEN logic) | Algorithm learns rules automatically from data |
| Computer follows fixed instructions | Computer improves with experience (more data) |
| Cannot handle unseen situations well | Generalizes to new, unseen inputs |

**Example:**
- **Traditional:** IF email contains "lottery winner" AND "click here" THEN mark as spam.
- **ML:** Show the algorithm 1 million spam and non-spam emails. It learns its own rules automatically.

---

### Q3: What are the three main types of Machine Learning?
**Answer:**
1. **Supervised Learning** – You provide labeled data (input + correct answer).
  - *Example:* Email (input) → Spam or Not Spam (label)
  - *Algorithms:* Linear Regression, Logistic Regression, Decision Trees, Random Forest

2. **Unsupervised Learning** – You only provide inputs; the algorithm finds hidden patterns itself.
  - *Example:* Customer shopping data → Groups customers by behavior automatically
  - *Algorithms:* K-Means Clustering, PCA

3. **Reinforcement Learning** – An agent learns by trial and error to maximize a reward.
  - *Example:* A robot learns to walk by trying different movements and getting rewarded for not falling.

---

### Q4: What is the difference between Classification and Regression?
**Answer:**
- **Classification**: Predicts a **category / class label**.
 - *Examples:* Spam or Not Spam, Cat or Dog, Fraud or Legitimate
- **Regression**: Predicts a **continuous numerical value**.
 - *Examples:* House price (₹52,00,000), temperature tomorrow (34°C), stock price

```python
# Classification Example (Predicts 0 or 1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() # Spam Classifier

# Regression Example (Predicts any number)
from sklearn.linear_model import LinearRegression
model = LinearRegression()  # House Price Predictor
```

---

## Section 2: Common Algorithms

### Q5: What is Linear Regression? Give an example.
**Answer:**
Linear Regression finds the best straight line that fits your data to predict a continuous value.

**Formula:** `Prediction = m × (Input) + b` 
- `m` = slope (how steep the line is)
- `b` = y-intercept (where the line crosses y-axis)

**Example:** Predict a student's exam score based on hours studied.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
hours = np.array([[1], [2], [3], [4], [5]])
scores = np.array([50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(hours, scores)

# Predict score for 6 hours
print(model.predict([[6]])) # Output: ~100
```

---

### Q6: What is Logistic Regression? Why is it used for Classification despite having "Regression" in its name?
**Answer:**
Logistic Regression predicts the **probability** that something belongs to a class (0 or 1). Despite the name "Regression," it is used for **Classification** because its output is always between 0 and 1 (a probability), and we apply a threshold (e.g., > 0.5 → Class 1).

```python
from sklearn.linear_model import LogisticRegression

# Predict if a student passes (1) or fails (0) based on study hours
X = [[2], [4], [6], [8]] # hours studied
y = [0, 0, 1, 1]     # pass/fail

model = LogisticRegression()
model.fit(X, y)
print(model.predict([[5]]))   # → [1] (Pass)
print(model.predict_proba([[5]]))# → Probability of failing and passing
```

---

### Q7: What is a Decision Tree?
**Answer:**
A Decision Tree is a flowchart-like model that makes decisions by asking a series of yes/no questions about the data features, splitting data at each step.

```
         [ Hours Studied > 4? ]
        /           \
       No (≤4)        Yes (>4)
        |            |
  [ Slept 8 hrs? ]     → Predict: PASS 
    /     \
   No      Yes
    |       |
 → FAIL    → PASS 
```

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### Q8: What is a Random Forest? How is it better than a single Decision Tree?
**Answer:**
Random Forest creates a **large collection (forest) of many Decision Trees**, each trained on a random subset of data. The final prediction is based on the **majority vote** of all trees.

```
     Data
     ↓
 ┌─────────────────┐
 │ Tree 1 → SPAM  │
 │ Tree 2 → SPAM  │ → Majority Vote → SPAM 
 │ Tree 3 → NOT  │
 └─────────────────┘
```

**Why better than a single tree?**
- A single tree can memorize training data (overfit).
- Random Forest averages many trees, reducing errors and being more robust.

---

## Section 3: Model Evaluation Metrics

### Q9: What is Overfitting and Underfitting?
**Answer:**
- **Overfitting**: The model memorizes the training data too well — including noise. Works great on training data but performs poorly on new/unseen data.
- **Underfitting**: The model is too simple — it fails to capture the patterns even in training data. Both training and test performance are poor.

```
Underfitting: Training = 60%, Test = 58%  (Both bad)
Good Fit:   Training = 92%, Test = 90%  (Both good)
Overfitting:  Training = 99%, Test = 55%  (Training great, Test terrible)
```

**Fix for Overfitting:** Get more data, simplify the model, use regularization.
**Fix for Underfitting:** Use a more complex model, add more features.

---

### Q10: What is a Confusion Matrix? Explain TP, TN, FP, FN.
**Answer:**
A Confusion Matrix is a table that summarizes how well a classification model performed.

```
           Predicted: SPAM | Predicted: NOT SPAM
Actual: SPAM  |  TP (True Positive) | FN (False Negative)
Actual: NOT SPAM|  FP (False Positive)| TN (True Negative)
```

- **TP (True Positive)**: Model said Spam, it IS Spam. Correct!
- **TN (True Negative)**: Model said Not Spam, it IS NOT Spam. Correct!
- **FP (False Positive)**: Model said Spam, but it IS NOT Spam. Wrong! (Important email sent to spam)
- **FN (False Negative)**: Model said Not Spam, but it IS Spam. Wrong! (Spam got through)

---

### Q11: What is Accuracy, Precision, Recall, and F1-Score?
**Answer:**
```
Accuracy = (TP + TN) / Total         → Overall correct predictions
Precision = TP / (TP + FP)           → Of all PREDICTED spam, how many were actually spam?
Recall  = TP / (TP + FN)           → Of all ACTUAL spam, how many did we catch?
F1-Score = 2 × (Precision × Recall) / (Precision + Recall) → Balance of Precision and Recall
```

**When to use what:**
- **Cancer detection:** Maximize **Recall** (don't miss any sick patients, even if we flag some healthy ones).
- **Spam filter:** Maximize **Precision** (don't put real emails in spam, even if some spam slips through).
- **Class imbalance:** Use **F1-Score** instead of accuracy.

---

### Q12: What is Train-Test Split? Why do we split the data?
**Answer:**
We split data into training and testing sets to evaluate how well the model generalizes to **new, unseen data**.
- **Training Set (~80%):** Used to teach the model patterns.
- **Test Set (~20%):** Used to measure performance on data the model has never seen.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```
