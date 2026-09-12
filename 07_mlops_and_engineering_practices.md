# Module 07: MLOps & Engineering Practices for Freshers

This module covers the practical engineering skills AI/ML freshers are expected to know for deploying and managing machine learning models in real-world projects.

---

## Section 1: What is MLOps?

### Q1: What is MLOps? Why is it important?
**Answer:**
**MLOps (Machine Learning Operations)** is the set of practices that combines Machine Learning with software engineering and DevOps principles to **deploy, monitor, and maintain ML models in production** reliably.

**The Problem MLOps solves:**
- A data scientist builds a model in a Jupyter notebook on their laptop → 90% accuracy!
- Then they hand it to the engineering team → It breaks on the production server.
- Or the model works initially but becomes inaccurate 3 months later because the real-world data has changed.

**MLOps ensures:**
- Models are reproducible (same code + same data = same results every time).
- Models can be deployed and scaled automatically.
- Model performance is tracked and monitored continuously.
- When something breaks, the team knows quickly.

---

### Q2: What is the difference between a Data Scientist and an ML Engineer?
**Answer:**

| | Data Scientist | ML Engineer |
|---|---|---|
| **Primary Focus** | Experiment: finding the best model | Production: deploying and maintaining the model |
| **Tools** | Jupyter Notebooks, Pandas, Matplotlib | Docker, APIs, cloud platforms, CI/CD pipelines |
| **Deliverable** | Trained model / insights | Reliable, scalable production service |
| **Skills** | Statistics, data exploration, modeling | Software engineering, DevOps, systems design |

**An AI/ML Trainee Engineer** sits between these two roles — must understand both modeling and deployment.

---

## 🐍 Section 2: Model Deployment Basics

### Q3: What is an API? How do you serve an ML model as an API?
**Answer:**
An **API (Application Programming Interface)** is a way for different software systems to communicate. When you deploy an ML model as an API:
- Other apps can send data to your model over the internet.
- Your model processes it and sends back predictions.

**FastAPI** is the most popular Python framework for serving ML models:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("spam_classifier.joblib") # Load your trained model

class EmailRequest(BaseModel):
  text: str

@app.post("/predict")
def predict_spam(request: EmailRequest):
  # Simple feature: word count (in real life, use TF-IDF or embeddings)
  features = [[len(request.text.split())]]
  prediction = model.predict(features)[0]
  label = "SPAM" if prediction == 1 else "NOT SPAM"
  return {"label": label}

# Run with: uvicorn main:app --reload
# Test at: http://localhost:8000/docs
```

---

### Q4: What is a REST API? What are GET, POST, PUT, DELETE?
**Answer:**
REST (Representational State Transfer) is the standard way to design APIs on the web. HTTP methods define what kind of action you're performing:

| HTTP Method | Action | ML Example |
|---|---|---|
| **GET** | Retrieve/read data | `GET /model/status` → Check if model is loaded |
| **POST** | Create/submit new data | `POST /predict` → Send input, receive prediction |
| **PUT** | Update existing data | `PUT /model/update` → Replace the model with a new version |
| **DELETE** | Delete data | `DELETE /model/cache` → Clear prediction cache |

```python
# Testing your API with Python requests
import requests

response = requests.post(
  "http://localhost:8000/predict",
  json={"text": "Congratulations! You won a free iPhone. Click now!"}
)
print(response.json()) # {"label": "SPAM"}
```

---

## 🐳 Section 3: Docker for ML

### Q5: What is Docker? Why is it used in ML deployments?
**Answer:**
Docker is a tool that packages your application and all its dependencies (Python version, libraries, system files) into a single portable unit called a **Container**.

**The "Works on my machine" problem:**
```
Developer: "My model works perfectly! 98% accuracy!"
    ↓
Server:  "ModuleNotFoundError: No module named 'sklearn'"
```

Docker solves this by saying: *"If it works in my Docker container, it will work on ANY machine running Docker — your laptop, a server, or the cloud."*

**Simple Docker workflow for ML:**
```dockerfile
# Dockerfile — Recipe for your container
FROM python:3.11-slim     # Start from Python image

WORKDIR /app          # Set working directory

COPY requirements.txt .    # Copy dependencies list
RUN pip install -r requirements.txt # Install them

COPY . .            # Copy your code + model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] # Start the server
```

```bash
# Build and run your Docker container
docker build -t my-ml-model .
docker run -p 8000:8000 my-ml-model
```

---

### Q6: What is the difference between an Image and a Container in Docker?
**Answer:**
- **Docker Image**: A read-only blueprint/recipe (like a class in Python). Built once.
- **Docker Container**: A running instance of an image (like an object from a class). You can run many containers from one image.

```
Image (Blueprint):  "python-3.11 + scikit-learn + FastAPI + my model code"
  ↓ docker run
Container (Running): An active instance serving predictions on port 8000
```

---

## Section 4: Version Control & Experiment Tracking

### Q7: Why is Git important in ML projects?
**Answer:**
Git is a version control system that tracks changes to your code files over time. It is essential in ML projects to:
- Collaborate with team members without overwriting each other's work.
- Track every change with a message: *"Changed learning rate from 0.01 to 0.001, accuracy improved 3%"*.
- Roll back to previous versions if something breaks.
- Work on different features/experiments in separate **branches** without affecting the main code.

```bash
# Common Git commands every ML fresher must know
git init             # Initialize git in your project
git add train.py model.py    # Stage files for commit
git commit -m "Add early stopping to training loop" # Save snapshot
git push origin main       # Upload to GitHub

git checkout -b experiment-new-features # Create new branch
git merge experiment-new-features    # Merge back when ready
```

---

### Q8: What is MLflow? Why do ML teams use it?
**Answer:**
**MLflow** is an open-source tool for tracking machine learning experiments. When you train many models trying different settings, it's hard to remember which configuration gave the best result.

MLflow records:
- **Parameters**: What settings you used (learning_rate=0.001, epochs=50)
- **Metrics**: Performance results (accuracy=0.91, F1=0.89)
- **Artifacts**: The saved model file, plots, confusion matrix

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
  # Log parameters
  mlflow.log_param("n_estimators", 100)
  mlflow.log_param("max_depth", 5)

  # Train model
  model = RandomForestClassifier(n_estimators=100, max_depth=5)
  model.fit(X_train, y_train)

  # Log metric
  acc = accuracy_score(y_test, model.predict(X_test))
  mlflow.log_metric("accuracy", acc)

  # Save model
  mlflow.sklearn.log_model(model, "model")

# View all experiments: mlflow ui (opens web dashboard)
```

---

## Section 5: Data Quality & Model Monitoring

### Q9: What is Data Drift? Why does it cause models to fail over time?
**Answer:**
**Data Drift** happens when the real-world data that a model receives in production **changes significantly** from the data it was trained on. This causes model accuracy to degrade silently over time.

**Real-world example:**
- A loan approval model was trained on customer data from 2020-2022.
- In 2024, a new generation of younger borrowers applies → Their income patterns, spending behavior, and credit history patterns are completely different.
- The model keeps running but gives incorrect predictions without anyone realizing!

**Types of Drift:**
1. **Data Drift (Covariate Shift):** The input features $X$ change (new customer demographics).
2. **Concept Drift:** The relationship between input and output changes (e.g., COVID changed what predicts loan default).
3. **Label Drift:** The distribution of the target labels $y$ changes (e.g., fraud rates increase from 0.1% to 5%).

**How to detect it:**
- Monitor statistical summaries (mean, std, min/max) of incoming features daily.
- Alert when significant deviation is detected.

---

### Q10: What is CI/CD? How does it apply to ML projects?
**Answer:**
**CI/CD (Continuous Integration / Continuous Deployment)** is the practice of automatically testing and deploying code changes, ensuring quality at every step.

In ML:
- **CI (Continuous Integration)**: Every time a data scientist pushes new code or a retrained model, automated tests run to check if accuracy hasn't dropped and the API still works correctly.
- **CD (Continuous Deployment)**: If all tests pass, the new model version is automatically deployed to production — no manual steps.

```
Code Push to GitHub
    ↓
[CI Pipeline Runs]:
 1. Run unit tests
 2. Train model on fresh data
 3. Evaluate: accuracy > 90%? 
 4. Test API endpoints 
    ↓
[CD Pipeline]:
 5. Build Docker image
 6. Push to container registry
 7. Deploy to production server
 8. Send Slack notification 
```
