# Coding Interview Questions for AI/ML Freshers

While AI/ML interviews test heavy theory and system design, almost every technical round begins with a **hands-on coding screening**. This guide covers the most frequent live-coding challenges across Python, NumPy, Pandas, and custom ML algorithm implementations.

---

## 🐍 Section 1: Python Core & Algorithmic Thinking

### Q1: Two Sum Problem (Optimized $O(N)$)
**Problem:** Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    # Use a hash map to store value -> index for O(1) lookups
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Test
print(two_sum([2, 7, 11, 15], 9)) # Output: [0, 1]
```
- **Time Complexity:** $O(N)$ — single pass through the list.
- **Space Complexity:** $O(N)$ — storing seen values in the dictionary.
- **Common Mistake:** Writing nested loops ($O(N^2)$). Always prioritize dictionary lookups for two-element sum questions.

---

### Q2: Reverse String In-Place / Two-Pointer Technique
**Problem:** Reverse a list of characters in-place without using Python's `[::-1]` slice or built-in `.reverse()`.

```python
def reverse_characters(s: list[str]) -> None:
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

chars = ['h', 'e', 'l', 'l', 'o']
reverse_characters(chars)
print(chars) # Output: ['o', 'l', 'l', 'e', 'h']
```
- **Time Complexity:** $O(N)$
- **Space Complexity:** $O(1)$ — zero auxiliary memory allocated.

---

## 🐼 Section 2: Pandas Data Wrangling (Take-Home & Screenings)

### Q3: Imputing Missing Values by Group
**Problem:** You have employee data with missing salaries (`NaN`). Fill missing values with the **median salary of their respective department**.

```python
import pandas as pd
import numpy as np

data = {
    'Department': ['Engineering', 'Engineering', 'HR', 'HR', 'Engineering', 'HR'],
    'Salary': [120000, np.nan, 70000, np.nan, 140000, 65000]
}
df = pd.DataFrame(data)

# Impute using transform to maintain DataFrame shape
df['Salary'] = df['Salary'].fillna(df.groupby('Department')['Salary'].transform('median'))
print(df)
```
- **Why `transform()`:** Unlike `.agg()`, `.transform()` returns a Series with the exact same index as the original DataFrame, allowing seamless vectorized assignment.

---

### Q4: Multi-Metric Aggregation & Filtering
**Problem:** Calculate total sales and average transaction amount per `Region`, keeping only regions with total sales $> \$50,000$.

```python
sales_data = {
    'Region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'Amount': [30000, 25000, 15000, 10000, 60000, 40000]
}
df_sales = pd.DataFrame(sales_data)

summary = (
    df_sales.groupby('Region')['Amount']
    .agg(total_sales='sum', avg_sales='mean')
    .query('total_sales > 50000')
    .reset_index()
)
print(summary)
```

---

## 📐 Section 3: Machine Learning Algorithms from Scratch (NumPy)

### Q5: Implement Vectorized Cosine Similarity without Scikit-Learn
**Problem:** Compute cosine similarity between matrix $A$ ($N \times D$) and vector $B$ ($D$).

$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$

```python
import numpy as np

def cosine_similarity_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Dot product along feature axis
    dot_product = np.dot(A, B)
    # L2 norms
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B)
    
    # Add epsilon to denominator to prevent division by zero
    epsilon = 1e-9
    return dot_product / (norm_A * norm_B + epsilon)

# Test
A = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
B = np.array([1.0, 2.0])
print("Similarities:", cosine_similarity_vectorized(A, B))
```

---

### Q6: Implement Numerically Stable Softmax
**Problem:** Write a Softmax function in NumPy that does not crash or overflow with large input values (e.g., logits = `[1000, 1001, 1002]`).

```python
def stable_softmax(logits: np.ndarray) -> np.ndarray:
    # Numerical stability trick: subtract max value to prevent exp() overflow
    shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

large_logits = np.array([1000.0, 1001.0, 1002.0])
# Standard np.exp(large_logits) results in [inf, inf, inf] and NaN
print("Stable Probabilities:", stable_softmax(large_logits))
# Output: [0.09003057, 0.24472847, 0.66524096]
```
> **Interview Gold:** Subtracting `np.max(x)` works mathematically because $\frac{e^{x_i - c}}{\sum e^{x_j - c}} = \frac{e^{x_i} / e^c}{\sum (e^{x_j} / e^c)} = \frac{e^{x_i}}{\sum e^{x_j}}$, preventing floating-point overflow.

---

## 💡 Top Coding Interview Tips
1. **Always State Time & Space Complexity:** State your Big-$O$ before writing code and confirm it after you finish.
2. **Handle Edge Cases Out Loud:** Check for empty arrays, division by zero, duplicate keys, and negative numbers.
3. **Avoid Python Loops for Data:** Whenever possible in Pandas/NumPy, use vectorized operations instead of `.iterrows()` or `.apply()`.
