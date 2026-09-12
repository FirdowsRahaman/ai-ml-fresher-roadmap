# Module 04: Deep Learning Fundamentals for Freshers

This module covers the basics of Deep Learning — how neural networks work, why they are powerful, and common terms you'll hear in interviews.

---

## Section 1: What is Deep Learning?

### Q1: What is Deep Learning? How is it different from Machine Learning?
**Answer:**
- **Machine Learning (ML):** You manually pick features from data (e.g., "use pixel brightness, edge angles, color histogram") and feed them into an algorithm.
- **Deep Learning (DL):** A neural network automatically learns which features are important directly from raw data (images, text, audio) — no manual feature engineering needed.

```
Machine Learning Pipeline:
Raw Data → [You extract features] → ML Algorithm → Output

Deep Learning Pipeline:
Raw Data → [Neural Network learns features automatically] → Output
```

**Deep Learning requires:** Large amounts of data and powerful GPUs to train.

---

### Q2: What is a Neural Network? What is a Neuron?
**Answer:**
A Neural Network is a mathematical model loosely inspired by the human brain. It consists of layers of interconnected **neurons** (also called **nodes** or **perceptrons**).

Each neuron:
1. Receives input values (numbers).
2. Multiplies each input by a **weight** (importance score).
3. Adds a **bias** value.
4. Passes the result through an **activation function** to decide what signal to output.

```
     Input 1 (x₁) ──── weight₁ ──┐
     Input 2 (x₂) ──── weight₂ ──┤──[ Sum + Bias ]──[ Activation ]──→ Output
     Input 3 (x₃) ──── weight₃ ──┘
```

```python
import numpy as np

def neuron(inputs, weights, bias):
  raw = np.dot(inputs, weights) + bias # Weighted sum
  return max(0, raw) # ReLU activation

inputs = np.array([0.5, 0.8, 0.2])
weights = np.array([0.4, 0.6, 0.9])
bias  = 0.1

print(neuron(inputs, weights, bias)) # 0.7 → neuron fires
```

---

### Q3: What are the layers in a Neural Network?
**Answer:**
A neural network is organized into three types of layers:

```
[ Input Layer ] → [ Hidden Layer 1 ] → [ Hidden Layer 2 ] → [ Output Layer ]
 (Raw data in)   (Learns features)   (Learns complex)  (Prediction out)
```

- **Input Layer:** Accepts the raw data (e.g., pixel values of an image).
- **Hidden Layers:** Process and transform data. The "deep" in Deep Learning refers to having many hidden layers.
- **Output Layer:** Produces the final prediction (e.g., 1 neuron for regression, 10 neurons for 10-class classification).

---

## ⚡ Section 2: Activation Functions

### Q4: What is an Activation Function? Why is it needed?
**Answer:**
Without an activation function, stacking multiple layers would still only produce a **linear output** (same as one layer). Activation functions introduce **non-linearity**, allowing the network to learn complex patterns.

Think of it this way: without activation functions, a deep network is just a fancier version of linear regression.

---

### Q5: What are the most common Activation Functions?
**Answer:**

| Activation | Formula | Output Range | When Used |
|---|---|---|---|
| **Sigmoid** | 1 / (1 + e⁻ˣ) | (0, 1) | Output layer for binary classification |
| **Softmax** | eˣⁱ / Σeˣ | (0, 1), sums to 1 | Output layer for multi-class classification |
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers — most commonly used today |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | (-1, 1) | Sometimes used in hidden layers |

```python
import numpy as np

def relu(x):
  return np.maximum(0, x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.array([-2, -1, 0, 1, 2])
print("ReLU:  ", relu(x))   # [0, 0, 0, 1, 2]
print("Sigmoid:", sigmoid(x)) # [0.12, 0.27, 0.5, 0.73, 0.88]
```

---

## Section 3: Training a Neural Network

### Q6: What is a Loss Function (Cost Function)? Give examples.
**Answer:**
A **Loss Function** measures how wrong the model's predictions are. During training, the goal is to minimize the loss.

| Problem Type | Loss Function | When to Use |
|---|---|---|
| **Regression** | Mean Squared Error (MSE) | Predicting house prices, temperatures |
| **Binary Classification** | Binary Cross-Entropy | Spam vs Not Spam, Yes vs No |
| **Multi-Class Classification** | Categorical Cross-Entropy | Classifying 10 digit types (0-9) |

```python
import numpy as np

# MSE: average of squared differences
def mse(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)

y_true = np.array([3.0, 5.0, 2.0])
y_pred = np.array([2.5, 5.5, 2.0])

print(f"MSE: {mse(y_true, y_pred):.3f}") # 0.167
```

---

### Q7: What is Gradient Descent? Explain it simply.
**Answer:**
Gradient Descent is the algorithm that trains a neural network by **adjusting weights to minimize the loss function**, step by step.

**Simple Analogy:** Imagine you are blindfolded on a hilly landscape and want to reach the lowest point (minimum loss). Gradient Descent tells you: "the ground is sloping down to your left — take a step to the left." You repeat this until you reach the bottom.

```
Loss
 ^
 |  *
 |   *
 |    *  ← Each step reduces loss (gradient descent)
 |     *
 |      *
 +─────────────────> Epochs (Training iterations)
```

The size of each step is controlled by the **Learning Rate (η)**:
- Too high → Overshoot the minimum (model never converges).
- Too low → Takes forever to converge.

---

### Q8: What is Backpropagation?
**Answer:**
Backpropagation is the algorithm that calculates **how much each weight contributed to the error**, so we know which weights to adjust and by how much.

**Simple Steps:**
1. Do a **forward pass** → Data flows from input to output → Get a prediction.
2. Calculate the **loss** (how wrong the prediction was).
3. Do a **backward pass** (backpropagation) → Error signal flows backwards from output to input.
4. **Update each weight** to reduce the error using Gradient Descent.
5. Repeat for thousands of training examples.

---

### Q9: What is Overfitting in Deep Learning? How do you prevent it?
**Answer:**
A deep network with millions of parameters can memorize the training data completely — including noise. It performs very well on training data but fails on test data.

**Prevention Techniques:**
1. **Dropout** — Randomly "switches off" neurons during training. Forces the network to learn redundant representations and not rely on single neurons.
2. **Early Stopping** — Stop training when validation loss starts increasing.
3. **Data Augmentation** — Create more training data by rotating/flipping images.
4. **Regularization (L1/L2)** — Penalizes large weights.

```python
import torch.nn as nn

# Dropout example in PyTorch
model = nn.Sequential(
  nn.Linear(128, 64),
  nn.ReLU(),
  nn.Dropout(p=0.5),  # 50% neurons randomly disabled during training
  nn.Linear(64, 10)
)
```

---

### Q10: What is an Epoch, Batch Size, and Iteration?
**Answer:**
These three terms describe **how much data the model processes during training:**

- **Epoch**: One complete pass through the **entire training dataset**.
- **Batch Size**: Number of training samples processed in one step.
- **Iteration**: One weight update step (= 1 forward + 1 backward pass on one batch).

**Example:**
```
Dataset = 1,000 samples
Batch Size = 100

→ 1 Epoch = 1,000 / 100 = 10 Iterations
```

```python
# Typical training loop in PyTorch
for epoch in range(50):      # 50 epochs
  for batch in dataloader:   # Each batch = batch_size samples
    optimizer.zero_grad()
    outputs = model(batch['X'])
    loss = criterion(outputs, batch['y'])
    loss.backward()      # Backpropagation
    optimizer.step()     # Update weights
```

---

### Q11: What is the difference between SGD, Adam, and RMSprop optimizers?
**Answer:**
Optimizers are algorithms that decide **how to update the weights** during training.

| Optimizer | Description | When to Use |
|---|---|---|
| **SGD** (Stochastic Gradient Descent) | Basic optimizer. Updates weights using gradient of one random batch. | Simple problems; computer vision with momentum |
| **Adam** | Combines Momentum + adaptive learning rates. Very popular. | Default choice for most tasks — fast convergence |
| **RMSprop** | Adapts learning rate for each parameter individually. Good for RNNs | Sequence tasks, recurrent networks |

```python
import torch.optim as optim

# Adam is the most commonly used optimizer for freshers
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
