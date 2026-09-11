# Trainee AI/ML Engineer — Fresher Interview Preparation Guide

Welcome! This repository is a **complete interview preparation kit for freshers and trainees** applying for AI/ML Engineering roles. Everything here is explained simply — no unnecessary jargon.

---

## 📚 Complete Module Index

| # | Module | Topics | Difficulty |
|---|--------|---------|:----------:|
| 1 | [01. Python & Data Stack](./01_python_and_data_stack.md) | Python basics, Lists vs Tuples, NumPy arrays, Pandas DataFrames, handling NaN | 🟢 Beginner |
| 2 | [02. Math & Statistics](./02_math_and_statistics.md) | Vectors, Dot Product, Cosine Similarity, Mean/Median, Correlation, Probability, Normal Distribution, p-value | 🟢 Beginner |
| 3 | [03. Classical Machine Learning](./03_classical_machine_learning.md) | What is AI/ML, Supervised vs Unsupervised, Classification vs Regression, Linear/Logistic Regression, Decision Trees, Random Forest, Confusion Matrix, Accuracy/Precision/Recall/F1 | 🟢 Beginner |
| 4 | [04. Deep Learning Fundamentals](./04_deep_learning_fundamentals.md) | What is Deep Learning, Neurons, Layers, Activation Functions (ReLU/Sigmoid/Softmax), Loss Functions, Gradient Descent, Backpropagation, Dropout, Overfitting, Epoch/Batch/Iteration | 🟡 Intermediate |
| 5 | [05. NLP & Computer Vision](./05_nlp_and_computer_vision.md) | What is NLP, Tokens, Tokenization, Word Embeddings, Sentiment Analysis, NER, CNNs, Transfer Learning, Bag of Words, TF-IDF | 🟡 Intermediate |
| 6 | [06. Generative AI, LLMs & RAG](./06_genai_llms_and_rag.md) | What is Gen AI, LLMs (ChatGPT/Gemini/LLaMA), Tokens in LLMs, Prompting (Zero-shot/Few-shot/CoT), Hallucination, RAG, Vector Databases, Fine-Tuning vs RAG, Context Window, Quantization | 🟡 Must Know! |
| 7 | [07. MLOps & Engineering](./07_mlops_and_engineering_practices.md) | What is MLOps, APIs with FastAPI, Docker containers, Git basics, MLflow experiment tracking, Data Drift, CI/CD | 🟡 Intermediate |
| 8 | [08. Practical Scenarios & Coding](./08_practical_scenarios_and_coding.md) | How to answer "99% accuracy" trap, handling missing data, ML problem-solving approach, Coding: Linear Regression from scratch, Metrics from scratch, Pandas cleaning, LLM API call | 🟢 Practical |

---

## ⭐ Most Important Topics for Fresher AI/ML Interviews (2025-2026)

These are the questions every interviewer will ask:

### Tier 1 — You MUST know these (asked in 90%+ of interviews)
- [ ] What is AI, ML, and Deep Learning? What's the difference?
- [ ] What is Supervised vs Unsupervised vs Reinforcement Learning?
- [ ] What is Classification vs Regression? Give examples.
- [ ] What is Overfitting? How do you fix it?
- [ ] What is a Confusion Matrix? Explain TP, TN, FP, FN.
- [ ] What is Precision, Recall, and F1-Score? When to prefer which?
- [ ] What is a Neural Network? What is backpropagation?
- [ ] **What is Generative AI?** (New — very commonly asked in 2025)
- [ ] **What is an LLM? Give 3 examples.**
- [ ] **What is a Token?**
- [ ] **What is a Prompt? What is prompt engineering?**
- [ ] **What is Hallucination in LLMs?**
- [ ] **What is RAG?**

### Tier 2 — Good to know (asked in ~60% of interviews)
- [ ] What is Random Forest and why is it better than a Decision Tree?
- [ ] What is Train-Test Split? Why is it necessary?
- [ ] What is the difference between L1 and L2 Regularization?
- [ ] What is Transfer Learning?
- [ ] What is a CNN?
- [ ] What is Word Embedding / Word2Vec?
- [ ] What is a Vector Database?
- [ ] What is Docker and why is it used?

### Tier 3 — For advanced positions / internships at big companies
- [ ] What is the Bias-Variance Tradeoff?
- [ ] What is the Transformer architecture? What is Self-Attention?
- [ ] What is Fine-Tuning an LLM?
- [ ] What is MLflow or experiment tracking?
- [ ] What is Data Drift?

---

## 🗺️ Recommended 4-Week Study Plan

### Week 1 — Python + Math Foundations
- Days 1-3: [Module 01 - Python & Data Stack](./01_python_and_data_stack.md)
- Days 4-7: [Module 02 - Math & Statistics](./02_math_and_statistics.md)

### Week 2 — Classical Machine Learning
- Days 1-4: [Module 03 - Classical ML](./03_classical_machine_learning.md)
  - Focus: Understand Confusion Matrix, Precision, Recall, F1 — VERY commonly asked!
- Days 5-7: Practice building models with Scikit-learn on Kaggle beginner datasets

### Week 3 — Deep Learning + NLP/CV
- Days 1-4: [Module 04 - Deep Learning Fundamentals](./04_deep_learning_fundamentals.md)
- Days 5-7: [Module 05 - NLP & Computer Vision](./05_nlp_and_computer_vision.md)

### Week 4 — GenAI (Most In-Demand!) + MLOps + Practice
- Days 1-3: [Module 06 - Generative AI, LLMs & RAG](./06_genai_llms_and_rag.md) ← 🔥 HIGH PRIORITY
- Days 4-5: [Module 07 - MLOps & Engineering](./07_mlops_and_engineering_practices.md)
- Days 6-7: [Module 08 - Practical Scenarios & Coding](./08_practical_scenarios_and_coding.md)

---

## 💡 Interview Tips for Freshers

### How to structure your answer when you're not 100% sure:
1. **Start with what you know for certain** — don't go quiet.
2. **Give a simple analogy** — "It's like..." interviewers love this.
3. **Be honest about limits** — "I understand the concept but I haven't implemented it yet."
4. **Show curiosity** — "I've read about X but would love to learn more about it."

### Common Mistakes to Avoid:
- ❌ Saying "99% accuracy" without checking the data is balanced.
- ❌ Not splitting data before preprocessing (data leakage!).
- ❌ Using a complex model first instead of starting simple.
- ❌ Confusing "parameters" and "hyperparameters."
- ❌ Not asking clarifying questions when given an open-ended ML problem.

### Quick Revision Flashcards:
| Term | 1-Line Definition |
|------|-------------------|
| AI | Machines doing human-like tasks |
| ML | Machines learning rules from data |
| Deep Learning | ML using multi-layered neural networks |
| LLM | Large language model trained on billions of text tokens |
| Token | Smallest text unit processed by an LLM (~¾ of a word) |
| Prompt | Input instruction given to an LLM |
| Hallucination | LLM generating confident but false information |
| RAG | Adding external documents to LLM's context to reduce hallucination |
| Vector DB | Database that stores and searches text embeddings (meaning-based search) |
| Embedding | Converting text/images into a list of numbers that captures meaning |
| Overfitting | Model memorizes training data, fails on new data |
| Recall | Out of all actual positives, how many did we correctly detect? |
| Precision | Out of all predicted positives, how many were actually positive? |
| F1-Score | Harmonic mean of Precision and Recall |
| Transfer Learning | Reusing a pre-trained model for a new task |
| Gradient Descent | Optimization algorithm to minimize loss by updating weights iteratively |
| Epoch | One full pass through the entire training dataset |
| Batch Size | Number of samples processed before weights are updated |
| Dropout | Randomly deactivating neurons during training to prevent overfitting |
| Docker | Packaging your app + dependencies into a portable container |
