# Machine Learning Interview Questions

Foundational ML interviews focus heavily on whether you understand *why* a model works and *how* to evaluate it, rather than just knowing how to import `scikit-learn`.

## 1. Model Evaluation
**Q: What is the difference between Precision and Recall? When would you optimize for one over the other?**
*Answer:* 
- **Precision:** Out of all positive predictions, how many were actually positive? (Minimize False Positives). Optimize for precision in Spam Detection (you don't want an important email marked as spam).
- **Recall:** Out of all actual positives, how many did we find? (Minimize False Negatives). Optimize for recall in Cancer Detection (it's better to flag a false positive for review than to miss actual cancer).

**Q: Your model has 99% accuracy but is completely useless. How is this possible?**
*Answer:* Class imbalance. If 99% of your data is "Not Fraud" and 1% is "Fraud", a model that simply always guesses "Not Fraud" will be 99% accurate but will never catch any fraud. Use F1-score or AUC-ROC instead.

## 2. Overfitting & Underfitting (Bias-Variance Tradeoff)
**Q: How do you know if your model is overfitting, and how do you fix it?**
*Answer:* It performs very well on training data but poorly on validation/test data. 
*Fixes:* Add more data, simplify the model (e.g., reduce tree depth), use regularization (L1/L2), or use dropout (for neural networks).

**Q: What is the Bias-Variance Tradeoff?**
*Answer:* 
- High Bias (Underfitting): The model is too simple and misses the underlying patterns.
- High Variance (Overfitting): The model is too complex and memorizes the noise in the training data.
The goal is to find the sweet spot in the middle that minimizes total error.

## 3. Data Processing
**Q: What is Data Leakage?**
*Answer:* When information from outside the training dataset is used to create the model. A common mistake is applying scaling or imputation to the *entire* dataset before splitting it into train/test sets. The test set must be kept entirely unseen.

## 4. Recommendation & Ranking Systems (Top MNC Frequency)
**Q: How do you design a recommendation feed at scale (e.g., millions of items)?**
*Answer:* Use the two-stage architecture:
1. **Candidate Generation (Retrieval):** Fast ANN search using Two-Tower user/item embeddings to retrieve top ~300 candidates in <20ms.
2. **Ranking:** Deep neural network with cross-features (user history, time of day, item popularity) scoring candidates by predicted click probability.
3. **Re-Ranking:** Business logic, diversity, freshness, and deduplication.

**Q: How do you handle the Cold-Start problem in recommendation systems?**
*Answer:*
- **New Items:** Rely on content-based metadata (tags, categories, text embeddings) instead of interaction history; use multi-armed bandits ($\epsilon$-greedy) to explore and display new items to users.
- **New Users:** Prompt for initial interest onboarding tags, serve geo/demographic popular defaults, and quickly update embeddings dynamically after their first 3–5 clicks.

## 5. Production MLOps & Drift
**Q: You notice your production model's F1 score dropped by 10% over the last month, but input feature statistics (mean, variance) haven't changed. What is happening?**
*Answer:* This is **Concept Drift**. The statistical distribution of features $P(X)$ is identical, but the relationship between input features and target labels $P(Y|X)$ has changed (e.g., competitor launched a new product, or macroeconomic shift changed user buying intent). You must collect recent ground-truth labels and retrain the model.

**Q: What is Training-Serving Skew and how do you prevent it?**
*Answer:* Training-serving skew happens when feature transformations applied during model training differ slightly from transformations calculated in production at inference time (e.g., pandas vs SQL vs Go timestamp calculations). It is prevented by using a **Feature Store** (like Feast) that acts as a single source of truth with point-in-time correctness for both batch training and real-time serving.

## 6. ML Experimentation & System Design
**Q: How do you evaluate whether a new model should replace the production model?**
*Answer:* Offline evaluation metrics (AUC, F1, NDCG) are only the first gate. The true test is an **online A/B test**:
1. Randomize users into Control (Champion model) and Variant (Challenger model).
2. Monitor **Primary Success Metrics** (CTR, Conversion rate) for statistical significance ($p < 0.05$).
3. Strictly monitor **Guardrail Metrics** (p99 latency, memory utilization, API 5xx errors, user complaints) to ensure no regressions before full rollout.

