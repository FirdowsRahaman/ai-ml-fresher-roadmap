# Module 02: Math & Statistics for AI/ML Freshers

This module covers the basic math and statistics concepts freshers should know for AI/ML interviews. No heavy formulas — just clear explanations with examples.

---

## 📐 Section 1: Linear Algebra Basics

### Q1: What is a Vector? What is a Matrix?
**Answer:**
- A **Vector** is a 1D list of numbers (like a row or column of values). In ML, a vector usually represents one data point with multiple features.
- A **Matrix** is a 2D table of numbers (rows and columns). In ML, the entire dataset is often stored as a matrix.

```python
import numpy as np

# Vector (1D): one data point with 3 features
vector = np.array([5.1, 3.5, 1.4])   # e.g., height, weight, age

# Matrix (2D): 4 data points, 3 features each
matrix = np.array([[5.1, 3.5, 1.4],
                   [4.9, 3.0, 1.5],
                   [6.2, 3.4, 5.4],
                   [5.5, 2.6, 4.4]])

print(matrix.shape)   # (4, 3) → 4 rows, 3 columns
```

---

### Q2: What is a Dot Product? Where is it used in ML?
**Answer:**
The dot product multiplies corresponding elements of two vectors and sums the results.

```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32
```

**Use in ML:**
- Every prediction in Linear/Logistic Regression is a dot product: `prediction = weights · features + bias`
- Neural network layers compute dot products between inputs and weight matrices.

---

### Q3: What is Cosine Similarity? Why is it used for comparing text?
**Answer:**
Cosine Similarity measures **how similar the direction** of two vectors is, regardless of their magnitude (length). It produces a value between -1 and 1:
- **1.0** = Identical direction (very similar)
- **0.0** = 90° apart (unrelated)
- **-1.0** = Opposite direction (opposite meaning)

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Two text embeddings
king   = np.array([0.8, 0.3])
queen  = np.array([0.75, 0.35])
apple  = np.array([-0.5, 0.9])

print(f"king ↔ queen: {cosine_similarity(king, queen):.3f}")  # High similarity
print(f"king ↔ apple: {cosine_similarity(king, apple):.3f}")  # Low similarity
```

**Used In:** Text similarity search, RAG (finding relevant documents), recommendation systems.

---

## 📊 Section 2: Statistics Basics

### Q4: What is Mean, Median, and Mode? When do you use each?
**Answer:**
- **Mean** = Sum of all values ÷ count. Affected by outliers.
- **Median** = The middle value when sorted. Robust to outliers.
- **Mode** = The most frequently occurring value.

```python
import numpy as np
from scipy import stats

data = [10, 12, 13, 14, 200]   # 200 is an outlier

print(f"Mean:   {np.mean(data):.1f}")    # 49.8 — pulled up by outlier
print(f"Median: {np.median(data):.1f}")  # 13.0 — not affected!
print(f"Mode:   {stats.mode(data)[0]}")  # 10 (first value if all appear once)
```

> **Rule of Thumb:** Use **Median** for house prices, salaries, or any data with outliers. Use **Mean** for normally distributed data like test scores.

---

### Q5: What is Standard Deviation and Variance? What do they tell us?
**Answer:**
- **Variance** = Average of squared distances from the mean. Tells us how spread out the data is.
- **Standard Deviation (Std Dev)** = Square root of variance. In the same units as the data (easier to interpret).

```python
import numpy as np

scores_A = [85, 86, 87, 88, 89]   # All close → Low spread
scores_B = [60, 70, 87, 95, 100]  # Very varied → High spread

print(f"Class A Std Dev: {np.std(scores_A):.2f}")  # ~1.41
print(f"Class B Std Dev: {np.std(scores_B):.2f}")  # ~14.5
```

---

### Q6: What is Correlation? Difference between Positive and Negative Correlation?
**Answer:**
Correlation measures how strongly two variables move together. Range: -1 to +1.
- **Positive (+1)**: Both increase together (e.g., height and weight).
- **Negative (-1)**: One increases as the other decreases (e.g., car speed and travel time).
- **Zero (0)**: No linear relationship.

```python
import pandas as pd
df = pd.DataFrame({'study_hours': [1, 2, 3, 4, 5],
                   'exam_score':  [55, 65, 70, 80, 90]})

print(df.corr())   # Shows correlation matrix
# study_hours ↔ exam_score: ~0.99 (strong positive correlation)
```

---

### Q7: What is Probability? What is the difference between P(A) and P(A|B)?
**Answer:**
- **P(A)**: Plain probability of event A happening. E.g., probability that it rains today = 30%.
- **P(A|B)**: **Conditional Probability** — probability of A happening *given* that B has already happened.

**Example:** 
- P(Student passes exam) = 60%
- P(Student passes exam | Student studied 5 hours) = 90%

The `|` is read as "given that."

---

### Q8: What is a Normal Distribution (Bell Curve)?
**Answer:**
The Normal Distribution (also called Gaussian Distribution) is the most common distribution in nature and statistics. It forms a symmetric bell-shaped curve where:
- Most data clusters around the **mean (center)**.
- Data becomes rarer as you move away from the center.
- **68%** of data falls within 1 standard deviation of the mean.
- **95%** falls within 2 standard deviations.
- **99.7%** falls within 3 standard deviations.

**Examples:** Human heights, IQ scores, measurement errors.

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.random.normal(mean=170, scale=10, size=1000)  # Heights in cm
print(f"Mean:   {np.mean(data):.1f} cm")
print(f"Std Dev:{np.std(data):.1f} cm")
```

---

### Q9: What is a p-value in simple terms?
**Answer:**
The p-value tells you whether your result happened by **pure chance or by a real cause**.
- **Small p-value (< 0.05)**: The result is unlikely to be random. There IS a real effect. (**Statistically Significant**)
- **Large p-value (> 0.05)**: The result could easily be random. The effect may not be real.

**Simple Example:**
You test a new drug. Patients who took it recovered 2 days faster. Is it the drug, or just luck?
- p = 0.001 → Very unlikely to be luck → Drug actually works!
- p = 0.4   → Could easily be random → Drug may not work.
