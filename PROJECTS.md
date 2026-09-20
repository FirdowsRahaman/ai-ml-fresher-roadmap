# 🚀 Real-World AI/ML Portfolio Projects (Local & Cloud Tracks)

A generic "Iris classifier" or standard "PDF Chatbot" will **not** get you noticed by hiring managers. Companies look for projects that solve **domain-specific problems**, handle **real-world trade-offs (latency, cost, drift, imbalance, privacy)**, and demonstrate **system thinking**.

> 💡 **No Cloud Budget? No Problem!**  
> You do **not** need an expensive cloud account or paid API keys to build impressive portfolio projects. We organize these projects into two distinct tracks:
> 1. **Track 1: 100% Local Projects ($0 Cloud Cost)** — Run entirely on your personal laptop/CPU using free open-source tools (Ollama, DuckDB, ChromaDB, Local Google ADK `adk web`, Scikit-Learn, and MLflow).
> 2. **Track 2: Big Tech & Cloud Enterprise Projects** — Architecture blueprints mirroring how **AWS, Google Cloud (GCP), Meta, and Stripe** build scalable production systems.

---

# 💻 Track 1: 100% Local Projects (Zero Cloud / Zero Cost)

Build, run, and test these projects directly on your laptop without entering a credit card or paying cloud bills.

---

## 📌 Local Project 1: Fully Local Private RAG Assistant (Zero-Cloud / 100% Offline)
> **Real-World Problem:** Healthcare, legal, and defense organizations cannot upload proprietary documents to external cloud APIs (OpenAI/Anthropic) due to HIPAA, GDPR, and data sovereignty laws.  
> **Key Skills:** Ollama (Local LLM), ChromaDB / FAISS, Sentence-Transformers, FastAPI, Streamlit  
> **Cost:** **$0.00** (Runs 100% offline on consumer laptop CPU/Apple Silicon)

### 💡 The Problem
Build an enterprise document QA assistant that answers queries over private corporate policies and technical manuals with **zero data ever leaving the local machine**.

### 🏗️ Local Architecture & Implementation
```
[Local PDF Documents] ──► [pdfplumber / PyPDF]
                                │ (Recursive Character Chunking: 500 chars, 50 overlap)
                                ▼
                   [Local Sentence-Transformers]
                   (all-MiniLM-L6-v2 running locally on CPU)
                                │
                                ▼
                     [Local ChromaDB / FAISS]
                     (Stored on local disk in ./chroma_db)
                                │
     User Query ────────► [FastAPI Backend]
                                │
                                ▼
                   [Local Similarity Search]
                   (Top 4 chunks retrieved via Cosine Distance)
                                │
                                ▼
                     [Ollama Local Runtime]
                     (llama3.2:3b or mistral:7b running locally)
                                │
                                ▼
                     [Streamlit Web Interface]
                     (Real-time token streaming to user)
```

1. **Local LLM Engine:** Install and run **Ollama** locally (`ollama run llama3.2:3b`). It exposes a local HTTP endpoint at `http://localhost:11434`.
2. **Local Embeddings:** Use Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` locally via Python to generate 384-dimensional embeddings on CPU in milliseconds.
3. **Embedded Vector Store:** Store vectors locally in **ChromaDB** (`chromadb.PersistentClient(path="./local_db")`), requiring zero external server setup.
4. **Interactive UI & Serving:** Build a clean **Streamlit** or **FastAPI** web interface that streams tokens directly to the user at human reading speed.

### 📊 Interview Talking Points
- **Zero Data Leakage:** Highlight how this architecture complies with air-gapped security requirements.
- **Quantization:** Explain why a 4-bit quantized model (`q4_K_M`) fits comfortably in 4GB of laptop RAM without requiring a $2,000 GPU.
- **Latency:** Explain local inference throughput (achieving 25–35 tokens/sec on Apple M-series or modern Intel/AMD CPUs).

---

## 📌 Local Project 2: End-to-End Local MLOps Pipeline with DuckDB, MLflow & Drift Detection
> **Real-World Problem:** How to run a complete enterprise MLOps lifecycle (ETL, training, experiment tracking, drift detection, containerization) without paid cloud services like SageMaker or Snowflake.  
> **Key Skills:** DuckDB (Local SQL ETL), Scikit-Learn, XGBoost, Local MLflow, Evidently AI, Docker  
> **Cost:** **$0.00** (Runs entirely locally)

### 💡 The Problem
Predict customer churn or fraud on a large tabular dataset (100,000+ rows). You need automated feature extraction, model versioning, production monitoring, and containerized deployment on your machine.

### 🏗️ Local Architecture & Implementation
1. **Local High-Speed ETL (DuckDB):**
   - Query raw CSV/Parquet files directly using **DuckDB** in Python without setting up a heavy SQL database server:
     ```python
     import duckdb
     df = duckdb.query("""
         SELECT user_id, 
                avg(amount) as avg_tx, 
                count(*) as tx_count 
         FROM 'data/*.parquet' 
         GROUP BY user_id
     """).df()
     ```
2. **Local Experiment Tracking (MLflow UI):**
   - Run a self-hosted MLflow server backed by local SQLite:
     ```bash
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
     ```
   - Track parameters, training/validation metrics, and model artifacts (`mlflow.sklearn.log_model`).
3. **Automated Drift Monitoring (Evidently AI):**
   - Simulate a production data shift (e.g., month-over-month data).
   - Generate an automated **Evidently AI HTML Report** calculating the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** test on key features.
4. **Local Container Deployment (Docker + FastAPI):**
   - Package the best model from the local MLflow registry into a **Docker container** running a **FastAPI** `/predict` endpoint with auto-generated Swagger documentation at `localhost:8000/docs`.

### 📊 Interview Talking Points
- **Point-in-Time Correctness:** How you prevented data leakage during local feature engineering.
- **Model Governance:** How MLflow's local registry tracks champion vs. challenger models.
- **Drift Action Threshold:** Explain why $\text{PSI} \ge 0.25$ triggers a retrain alert in your pipeline.

---

## 📌 Local Project 3: Local Multi-Agent Task Assistant with Google ADK (`adk web`) & FastMCP
> **Real-World Problem:** Orchestrating specialized agent teams to inspect local files, run calculations, and query local databases with full visual debugging and zero cloud dependencies.  
> **Key Skills:** Google Agent Development Kit (`google-adk`), FastMCP, SQLite, Python File Tools  
> **Cost:** **$0.00**

### 💡 The Problem
Developers and analysts need an AI assistant that can inspect local code repositories, run unit tests, and query local databases autonomously without trusting sensitive company code to third-party SaaS tools.

### 🏗️ Architecture & Implementation
1. **Google ADK Orchestration:**
   - Define a code-first multi-agent system using `google-adk`:
     - **Root Agent:** Understands the user's objective and delegates tasks.
     - **DB Subagent:** Queries a local `sales.db` SQLite database using SQL tools.
     - **File Subagent:** Reads, analyzes, and formats local markdown/code files.
2. **Local MCP Tool Server (FastMCP):**
   - Implement a standalone Python **FastMCP Server** exposing secure local tools over standard input/output (stdio):
     ```python
     from mcp.server.fastmcp import FastMCP
     import sqlite3

     mcp = FastMCP("Local Data Server")

     @mcp.tool()
     def query_local_db(query: str) -> list:
         """Execute read-only SQL queries on local SQLite database."""
         conn = sqlite3.connect("company.db")
         return conn.execute(query).fetchall()
     ```
3. **Visual Trajectory Debugging (`adk web`):**
   - Launch the built-in graphical interface using:
     ```bash
     adk web
     ```
   - Inspect the visual execution tree: see every thought, tool call argument, and observation in real time.

### 📊 Interview Talking Points
- **ADK Subagent Pattern:** Why delegating to specialized subagents prevents prompt bloat and hallucination.
- **MCP Decoupling:** How MCP standardizes tools so any local agent can query files and databases cleanly.

---

# ☁️ Track 2: Big Tech & Cloud Enterprise Projects (AWS / GCP / Meta)

Architectural blueprints mirroring the actual production systems deployed by cloud giants and Tier-1 tech companies.

---

## ☁️ Project 4 (AWS Enterprise): Serverless Intelligent Document Processing (IDP) Pipeline
> **Big Tech Problem:** How AWS Bedrock & Enterprise Solutions teams process millions of unstructured invoices, clinical trials, and legal contracts with zero manual data entry.  
> **Key Cloud Stack:** Amazon Bedrock (Claude 3.5 Sonnet / Nova), AWS Textract, OpenSearch Serverless, AWS Comprehend (PII), AWS Lambda, EventBridge, DynamoDB

### 💡 The Problem
Process hundreds of thousands of multi-page invoices, insurance claims, and loan agreements daily. Standard OCR yields messy text, manual review is slow, and naive LLMs hallucinate numbers and leak PII (HIPAA/GDPR compliance risks).

### 🏗️ Cloud Architecture
- **Ingestion:** S3 event notification $\rightarrow$ **AWS EventBridge** $\rightarrow$ **AWS Lambda**.
- **OCR & PII Masking:** **AWS Textract** (retaining table/form layouts) + **AWS Comprehend** (redacting PII before model inference).
- **Extraction:** **Amazon Bedrock** (Claude 3.5 Sonnet with Pydantic JSON schema).
- **Audit & Search:** Structured entities saved to **DynamoDB**; semantic embeddings indexed in **Amazon OpenSearch Serverless** for multi-document cross-audit.

---

## ☁️ Project 5 (GCP Enterprise): Multimodal Video Intelligence & Semantic Search Engine
> **Big Tech Problem:** How Google Cloud (Vertex AI) enables media platforms, retail surveillance, and autonomous fleets to index and query petabytes of video using natural language.  
> **Key Cloud Stack:** Google Cloud Vertex AI, Gemini 2.5 Flash, Vertex AI Multimodal Embeddings, BigQuery Vector Search, Cloud Storage, Cloud Run

### 💡 The Problem
Query thousands of hours of video footage with natural language queries (*"Find the exact timestamp where the delivery driver places the package behind the red pillar"*) without manual human tagging.

### 🏗️ Cloud Architecture
- **Frame Sampling:** Cloud Functions extract keyframes (1 fps) and audio transcripts from Google Cloud Storage.
- **Multimodal Embedding:** **Vertex AI Multimodal Embeddings API** maps frames and audio into a unified 1408-dimensional vector space.
- **BigQuery Vector Search:** Millions of multimodal vectors indexed directly in **BigQuery (HNSW index)**, enabling sub-100ms vector lookup without dedicated vector DB servers.
- **Temporal Verification:** **Gemini 2.5 Flash** reasons over top retrieved clips to output exact timestamp markers and bounding boxes.

---

## ☁️ Project 6 (Meta & Amazon): Two-Stage E-Commerce Recommendation & Ranking Engine
> **Big Tech Problem:** How Amazon and Meta rank billions of items in feeds and shopping search under strict 40ms latency constraints.  
> **Key Cloud Stack:** PyTorch, Two-Tower DNN, FAISS (HNSW), LightGBM (LambdaMART), Redis, FastAPI, AWS EC2

### 💡 The Problem
Recommending items from a catalog of 1,000,000+ products in real-time under a strict **40ms p99 latency SLA**.

### 🏗️ Cloud Architecture
- **Stage 1 (Retrieval):** Two-Tower Neural Network in PyTorch (User Tower + Item Tower) generating 64-dim embeddings; **FAISS HNSW** retrieves top 300 candidates in <10ms.
- **Stage 2 (Ranking):** **LightGBM LambdaMART** scores candidates on $P(\text{Click}) \times P(\text{Purchase})$.
- **Stage 3 (Diversity):** Maximal Marginal Relevance (MMR) prevents filter bubbles and guarantees merchant diversity.
- **Serving:** Dockerized FastAPI service deployed on AWS EC2, caching hot embeddings in Redis (total p99 latency: **27ms**).

---

## ☁️ Project 7 (AI Infrastructure): High-Throughput Enterprise LLM Serving with vLLM & Semantic Caching
> **Big Tech Problem:** How AI cloud infrastructure providers serve thousands of concurrent LLM requests under strict GPU memory limits and sub-second latency budgets.  
> **Key Cloud Stack:** vLLM (PagedAttention), Llama 3.1, Redis Semantic Cache, OpenTelemetry, Arize Phoenix, Docker, AWS g5.2xlarge

### 💡 The Problem
Serve 1,000+ customer queries/minute with p95 latency < 500ms while reducing commercial API costs by over 70%.

### 🏗️ Cloud Architecture
- **Serving:** Self-hosted open-weight model deployed via **vLLM (PagedAttention)**, boosting batch concurrency by 4x without KV cache fragmentation.
- **2-Tier Caching:** **Redis Semantic Cache** (serving identical questions in <15ms at $0 cost) + vLLM Prompt Caching (cutting TTFT by 70%).
- **Observability:** **OpenTelemetry** exporting traces to **Arize Phoenix** to monitor TTFT, TPOT, and GPU VRAM saturation.

---

## 📋 Resume Project Comparison Matrix

| Track | Project Name | Primary Stack | Cost | Best Suited For |
|---|---|---|---|---|
| **Local** | **Fully Local Private RAG** | Ollama, ChromaDB, Sentence-Transformers, Streamlit | **$0.00** | Privacy-first GenAI, Air-Gapped Systems, Healthcare/Legal |
| **Local** | **End-to-End Local MLOps** | DuckDB, XGBoost, MLflow UI, Evidently AI, Docker | **$0.00** | Production ML, Data Engineering, MLOps Roles |
| **Local** | **Local Multi-Agent Assistant** | Google ADK (`adk web`), FastMCP, SQLite, Python | **$0.00** | AI Agent Developer, Tool Engineering, Automation |
| **Cloud** | **Serverless IDP Pipeline** | AWS Bedrock, Textract, Comprehend, OpenSearch | Cloud | AWS Solutions Architect, Enterprise Cloud AI |
| **Cloud** | **Multimodal Video Intelligence** | GCP Vertex AI, BigQuery Vector Search, Gemini | Cloud | Computer Vision, Big Data AI, Media Platforms |
| **Cloud** | **Two-Stage Recommendation Engine** | Two-Tower PyTorch, FAISS HNSW, LightGBM, Redis | Cloud | Meta, Amazon, Netflix, E-Commerce Ranking |
| **Cloud** | **High-Throughput LLM Serving** | vLLM PagedAttention, Redis Semantic Cache, Phoenix | Cloud | AI Infrastructure Engineer, Platform Engineer |

---

## 💡 How to Present These Projects on Your Resume (CAR Framework)

When writing resume bullet points and answering interview questions, use the **CAR (Context, Action, Result)** format:
1. **Context:** State the problem and real-world constraint (*"Under HIPAA regulations..."*, *"Under a 40ms p99 SLA..."*).
2. **Action:** Explain what you built and why (*"Architected a fully local RAG pipeline using Ollama and ChromaDB, eliminating external API calls..."*).
3. **Result:** Provide measurable engineering outcomes (*"Achieved 100% data privacy with 28 tok/sec local inference speed on consumer hardware at $0 cloud cost"*).
