# Module 01: Python & Data Science Stack for Freshers

This module covers basic Python and data science questions that freshers are commonly asked in AI/ML interviews.

---

## Section 1: Python Basics

### Q1: What is Python? Why is it popular in AI/ML?
**Answer:**
Python is a high-level, easy-to-read programming language. It is popular in AI/ML because:
- It has simple, readable syntax (less code to write).
- It has powerful libraries like NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch.
- It has a huge community and thousands of free tutorials and resources.
- It runs on all major operating systems.

---

### Q2: What is the difference between a List and a Tuple in Python?
**Answer:**
| Feature | List `[ ]` | Tuple `( )` |
|---|---|---|
| **Mutability** | Mutable (can be changed after creation) | Immutable (cannot be changed) |
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` |
| **Speed** | Slightly slower | Slightly faster |
| **Use Case** | When you need to add/remove items | When data should not change (e.g., coordinates) |

```python
my_list = [1, 2, 3]
my_list[0] = 99  # Works fine

my_tuple = (1, 2, 3)
my_tuple[0] = 99 # TypeError: 'tuple' object does not support item assignment
```

---

### Q3: What is the difference between `==` and `is` in Python?
**Answer:**
- `==` checks if two variables have the **same value**.
- `is` checks if two variables point to the **exact same object in memory**.

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True → same values
print(a is b)  # False → different objects in memory

c = a
print(a is c)  # True → same object
```

---

### Q4: What are Python Generators? What is `yield`?
**Answer:**
A generator is a special function that produces values one at a time using the `yield` keyword instead of returning everything at once. This saves memory when dealing with large datasets.

```python
# Normal function — loads everything into memory at once
def get_all_numbers(n):
  return [i * 2 for i in range(n)] # Stores entire list

# Generator — produces values one-by-one (memory efficient!)
def generate_numbers(n):
  for i in range(n):
    yield i * 2 # Returns one value at a time, pauses until next call

gen = generate_numbers(1000000) # Takes almost zero memory
print(next(gen)) # 0
print(next(gen)) # 2
```

> **Why it matters in ML:** When loading millions of images or rows of data, generators allow you to process one batch at a time without running out of RAM.

---

### Q5: What is a Lambda function?
**Answer:**
A lambda is a small, anonymous (nameless) function written in a single line. Used for short, one-time operations.

```python
# Normal function
def square(x):
  return x ** 2

# Lambda equivalent
square = lambda x: x ** 2

# Commonly used with Pandas:
import pandas as pd
df = pd.DataFrame({'age': [25, 30, 35]})
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Senior')
```

---

### Q6: What is the difference between `append()` and `extend()` in a list?
**Answer:**
- `append()` adds a **single element** to the end of the list.
- `extend()` adds **all elements** from another iterable to the list.

```python
a = [1, 2, 3]
a.append([4, 5])
print(a) # [1, 2, 3, [4, 5]] ← The list [4,5] is added as ONE element

b = [1, 2, 3]
b.extend([4, 5])
print(b) # [1, 2, 3, 4, 5]  ← 4 and 5 are added individually
```

---

## Section 2: NumPy Basics

### Q7: What is NumPy? Why is it used in ML?
**Answer:**
NumPy (Numerical Python) is a library that provides fast mathematical operations on large arrays and matrices. Machine learning involves processing thousands to millions of numbers simultaneously. NumPy operations run in fast C code under the hood — much faster than native Python loops.

```python
import numpy as np

# Without NumPy (slow Python loop)
result = [x * 2 for x in range(1_000_000)] # Slow

# With NumPy (fast)
arr = np.arange(1_000_000)
result = arr * 2 # About 30x faster
```

---

### Q8: What is the shape of a NumPy array? How do you check it?
**Answer:**
Shape describes the dimensions of an array — like rows and columns in a table.

```python
import numpy as np

a = np.array([1, 2, 3])
print(a.shape) # (3,) → 1D array with 3 elements

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape) # (2, 3) → 2 rows, 3 columns

# Reshaping
c = b.reshape(3, 2) # Now 3 rows, 2 columns
print(c.shape) # (3, 2)
```

---

### Q9: What is Broadcasting in NumPy?
**Answer:**
Broadcasting allows NumPy to perform arithmetic operations between arrays of different shapes without creating copies of data.

```python
import numpy as np

arr = np.array([[1, 2, 3],
        [4, 5, 6]])   # Shape (2, 3)

row = np.array([10, 20, 30])  # Shape (3,)

result = arr + row       # Broadcasting adds row to EACH row of arr
# [[11, 22, 33],
# [14, 25, 36]]
```

---

## Section 3: Pandas Basics

### Q10: What is Pandas? What is a DataFrame?
**Answer:**
Pandas is a Python library for working with structured data (think Excel tables in Python). A **DataFrame** is the core data structure — a 2D table with rows and labeled columns.

```python
import pandas as pd

data = {
  'Name': ['Alice', 'Bob', 'Charlie'],
  'Age':  [25, 30, 28],
  'Score': [85.0, 90.5, 78.0]
}

df = pd.DataFrame(data)
print(df)
#   Name Age Score
# 0  Alice  25  85.0
# 1   Bob  30  90.5
# 2 Charlie  28  78.0
```

---

### Q11: How do you handle missing values in Pandas?
**Answer:**
Missing values appear as `NaN` (Not a Number). You can detect and handle them:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'],
          'score': [85, np.nan, 90]})

# Detect missing values
print(df.isnull())     # Shows True where NaN exists
print(df.isnull().sum())  # Count of NaN per column

# Option 1: Drop rows with NaN
df_dropped = df.dropna()

# Option 2: Fill with a value (mean for numerical columns)
df['score'].fillna(df['score'].mean(), inplace=True)

# Option 3: Fill with 0 or a specific value
df['score'].fillna(0, inplace=True)
```

---

### Q12: What is the difference between `loc` and `iloc`?
**Answer:**
- `loc` → Access rows/columns by **label (name)**.
- `iloc` → Access rows/columns by **integer position (index number)**.

```python
df = pd.DataFrame({'score': [85, 90, 78]}, index=['Alice', 'Bob', 'Charlie'])

# loc — by label
print(df.loc['Alice'])   # Gets row for Alice

# iloc — by position
print(df.iloc[0])     # Gets the first row (index 0)

# Slicing
print(df.loc['Alice':'Bob'])  # Rows from Alice to Bob (inclusive)
print(df.iloc[0:2])      # Rows 0 and 1 (exclusive end)
```

---

## Section 4: SQL & Data Engineering Basics

### Q13: Why is SQL important for Machine Learning?
**Answer:**
Before you can analyze data with Pandas or train a model with Scikit-Learn, you have to get the data out of the company's database. SQL (Structured Query Language) is the standard language for extracting and manipulating data in relational databases.
- Most real-world ML datasets aren't CSV files on a laptop; they are tables in a database (like PostgreSQL, MySQL, or Snowflake).
- Performing basic filtering, joining, and aggregation in SQL is often much faster and more memory-efficient than loading raw data into Python.

### Q14: Explain the difference between INNER JOIN, LEFT JOIN, and RIGHT JOIN.
**Answer:**
Joins are used to combine rows from two or more tables based on a related column.
- **INNER JOIN:** Returns only the records that have matching values in both tables.
- **LEFT JOIN:** Returns all records from the left table, and the matched records from the right table. If there is no match, the result is NULL on the right side.
- **RIGHT JOIN:** Returns all records from the right table, and the matched records from the left table.

### Q15: What is an ETL Pipeline?
**Answer:**
ETL stands for **Extract, Transform, Load**. It is the core concept of Data Engineering.
- **Extract:** Pull data from various sources (SQL databases, APIs, logs).
- **Transform:** Clean the data, handle missing values, format dates, and join tables.
- **Load:** Save the cleaned data into a Data Warehouse or a feature store where Data Scientists can easily access it for training models.
