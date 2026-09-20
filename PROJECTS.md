# 🚀 Real-World Big Tech & Cloud Enterprise AI/ML Projects

A generic "PDF Chatbot" or "Iris classification script" will **not** get you an interview at top tech companies or cloud giants (Google Cloud / GCP, AWS, Meta, Microsoft). Hiring managers look for projects that solve **complex enterprise problems**, use **production cloud services (Bedrock, Vertex AI, SageMaker, BigQuery)**, handle **real-world scale (latency SLAs, token budgets, drift, security)**, and demonstrate **system thinking**.

Here are 6 battle-tested, enterprise-grade portfolio projects mirroring the exact solutions built by **GCP, AWS, and Big Tech** engineering teams.

---

## ☁️ Project 1 (AWS Enterprise): Serverless Intelligent Document Processing (IDP) Pipeline
> **Big Tech Problem:** How AWS Bedrock & Enterprise Solutions teams process millions of unstructured invoices, clinical trials, and legal contracts with zero manual data entry.  
> **Target Roles:** AI Engineer, Cloud ML Engineer, Solutions Architect (AWS)  
> **Key Cloud Stack:** Amazon Bedrock (Claude 3.5 Sonnet / Nova), AWS Textract, OpenSearch Serverless, AWS Comprehend (PII), AWS Lambda, EventBridge, DynamoDB

### 💡 The Problem
Fortune 500 banks and healthcare providers process hundreds of thousands of multi-page invoices, insurance claims, and loan agreements daily. Standard OCR yields messy text, manual review is prohibitively slow, and naive LLM prompts hallucinate financial figures and leak PII (HIPAA/GDPR compliance risks).

### 🏗️ Cloud Architecture & Workflow
```
[User / ERP Upload] ──► [Amazon S3 Bucket]
                              │ (S3 Event Notification)
                              ▼
                      [AWS EventBridge]
                              │
                              ▼
                      [AWS Lambda Orchestrator]
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
  [AWS Textract Layout]                 [AWS Comprehend PII]
  (Extracts key-values & tables)         (Detects & masks SSN, names, cards)
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                 [Amazon Bedrock Extraction]
                 (Claude 3.5 Sonnet with Pydantic JSON schema)
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
  [Amazon DynamoDB]                     [OpenSearch Serverless]
  (Structured entities for ERP)          (Vector index for natural language audit)
```

1. **Event-Driven Ingestion:** Documents uploaded to S3 trigger an event via **EventBridge** to an **AWS Lambda** worker.
2. **Layout-Aware OCR:** **AWS Textract** extracts raw text while preserving tabular structures and key-value form fields.
3. **Automated PII Masking:** **AWS Comprehend** scans for HIPAA/PII entities (SSNs, medical IDs, credit card numbers) and redacts them prior to model inference.
4. **Structured JSON Extraction:** **Amazon Bedrock** runs Claude 3.5 Sonnet with strict JSON Schema constraints, transforming messy invoice tables into validated accounting payloads.
5. **Storage & Audit Index:** Structured payloads are saved to **DynamoDB** for downstream ERP consumption, while semantic embeddings are indexed in **Amazon OpenSearch Serverless** for multi-document cross-audit.

### 📊 Evaluation & Interview Talking Points
- **Throughput & Cost:** Serverless event-driven architecture handles 10,000+ docs/hour with zero idle server cost.
- **Accuracy:** Reached 99.2% entity extraction accuracy, reducing human-in-the-loop manual review by 88%.
- **Compliance:** Explain how automated PII redaction via Comprehend prevents regulatory breaches before data touches LLM context windows.

---

## ☁️ Project 2 (GCP Enterprise): Multimodal Video Intelligence & Semantic Search Engine
> **Big Tech Problem:** How Google Cloud (Vertex AI) enables media platforms, retail surveillance, and autonomous fleets to index and query petabytes of video using natural language.  
> **Target Roles:** Machine Learning Engineer, Computer Vision Engineer, Vertex AI Specialist  
> **Key Cloud Stack:** Google Cloud Vertex AI, Gemini 2.5 Flash, Vertex AI Multimodal Embeddings, BigQuery Vector Search, Cloud Storage, Cloud Run

### 💡 The Problem
Media broadcast networks and security providers have thousands of hours of video footage. Tagging video manually is impossible, and traditional metadata search only searches file names. Users need to search semantic events: *"Find the exact timestamp where the delivery driver places the package behind the red pillar."*

### 🏗️ Cloud Architecture & Workflow
```
[Raw Video Files] ──► [Google Cloud Storage (GCS)]
                              │
                              ▼
                    [Vertex AI Video Pipeline]
                    - Frame Sampling (1 fps) + Audio Whisper Transcripts
                              │
                              ▼
             [Vertex AI Multimodal Embeddings API]
             (Maps video frames + audio to 1408-dim vector space)
                              │
                              ▼
             [BigQuery Vector Search (HNSW Index)]
             (Indexes millions of multimodal vectors at petabyte scale)
                              │
     User Natural Language Query: "Delivery driver behind red pillar"
                              │
                              ▼
                    [BigQuery Vector Search]
                    (Sub-100ms Cosine Distance Lookup)
                              │
                              ▼
                    [Gemini 2.5 Flash]
                    (Synthesizes timestamp explanation & bounding box)
```

1. **Automated Ingestion & Frame Sampling:** Cloud Functions trigger on video uploads to GCS, extracting keyframes (1 fps) and synchronizing audio speech-to-text transcripts.
2. **Multimodal Vectorization:** Frames and transcripts are passed to the **Vertex AI Multimodal Embeddings API**, mapping visual content and audio into a unified 1408-dimensional embedding space.
3. **Petabyte-Scale Vector Indexing:** Embeddings are written to **BigQuery**, utilizing **BigQuery Vector Search (HNSW)** to perform sub-100ms approximate nearest neighbor search across billions of frame embeddings without spinning up dedicated vector databases.
4. **Temporal Grounding:** Top matching video clips are verified by **Gemini 2.5 Flash** using native spatio-temporal video reasoning to pinpoint exact second markers.

### 📊 Evaluation & Interview Talking Points
- **BigQuery Vector Search vs Dedicated Vector DBs:** Explain why BigQuery Vector Search eliminates data movement pipelines for enterprises already storing telemetry in Google Cloud.
- **Multimodal Alignment:** How visual embeddings and transcript text align in the same vector space.
- **Retrieval Speed:** Sub-100ms vector lookup across 50,000 hours of video content.

---

## 🤖 Project 3 (Google Ecosystem): Autonomous Operations & Support Copilot with Google ADK & MCP
> **Big Tech Problem:** How Google builds enterprise agent systems with code-first software engineering, tool integration, and safe subagent delegation.  
> **Target Roles:** AI Agent Engineer, AI Systems Developer, Full Stack AI Engineer  
> **Key Cloud Stack:** Google Agent Development Kit (`google-adk`), Gemini 2.5 Pro, Model Context Protocol (FastMCP), Cloud Run, Docker

### 💡 The Problem
Enterprise customer operations involve complex, multi-system workflows: checking order databases, issuing refunds, diagnosing shipping delays, and updating CRM records. Monolithic agents hallucinate or fail because they attempt to juggle too many tools simultaneously without state validation.

### 🏗️ Architecture & Implementation
1. **Google ADK Hierarchical Orchestration:**
   - Built with the code-first **Google Agent Development Kit (ADK)**.
   - **Root Agent (Supervisor):** Classifies customer intent, enforces company business policies, and delegates tasks to specialized subagents.
   - **Order Subagent:** Connects to ERP databases via SQL tools to verify order timestamps and delivery statuses.
   - **Refund Subagent:** Evaluates refund eligibility against business rules and prepares financial transactions.
2. **Standardized Tool Integration via Model Context Protocol (MCP):**
   - The agent communicates with backend systems through a **FastMCP Server** (exposing tools via JSON-RPC 2.0).
   - Tools are isolated: database mutation operations require explicit authorization tokens.
3. **Human-in-the-Loop (HITL) Gate:**
   - Any financial transaction over $100 or cancellation of high-tier subscriptions triggers an interrupt.
   - The state is preserved, and a human agent reviews the proposed action via the **`adk web`** dashboard before execution.
4. **Serverless Cloud Deployment:**
   - Packaged and deployed to **Google Cloud Run** using `adk deploy cloud_run`.

### 📊 Evaluation & Interview Talking Points
- **Google ADK vs LangGraph:** Explain how Google ADK applies software engineering discipline (clean Python classes, `adk web` local visual debugging, first-class subagent delegation) compared to raw state-graph wiring.
- **MCP Decoupling:** How MCP allowed swapping mock databases for production PostgreSQL without altering a single line of agent reasoning code.
- **Containment:** Successfully automated 74% of tier-1 customer inquiries while maintaining 0 unauthorized financial transactions.

---

## 🎯 Project 4 (Meta & Amazon): Two-Stage E-Commerce Recommendation & Ranking Engine
> **Big Tech Problem:** How Amazon and Meta rank billions of items in feeds and shopping search under strict 40ms latency constraints.  
> **Target Roles:** Machine Learning Engineer, Recommendation Systems Engineer, Applied Scientist  
> **Key Cloud Stack:** PyTorch, Two-Tower DNN, FAISS (HNSW), LightGBM (LambdaMART), Redis, FastAPI, AWS EC2

### 💡 The Problem
Recommending items from a catalog of 1,000,000+ products in real-time. Running a deep neural network across the full catalog causes unacceptable latency (>2,000ms), violating the **40ms p99 SLA**.

### 🏗️ Architecture & Implementation
1. **Stage 1 — Candidate Generation (Retrieval):**
   - Train a **Two-Tower Neural Network** in PyTorch:
     - *User Tower:* Encodes user demographics, recent interaction history, and device signals into a 64-dimensional embedding.
     - *Item Tower:* Encodes product title, category, price, and merchant metrics into a 64-dimensional embedding.
   - Store all item vectors in a **FAISS HNSW** index.
   - At inference, the User Tower generates an embedding in 4ms, and FAISS retrieves the **top 300 candidates in <10ms**.
2. **Stage 2 — Scoring & Ranking:**
   - Pass top 300 candidates to a **LightGBM LambdaMART** learning-to-rank model.
   - Features: Real-time context (time of day, cart value), cross-features (user CTR on item category), and discount percentage.
   - Predicts expected user engagement: $\text{Score} = P(\text{Click}) \times P(\text{Purchase}) \times \text{Item Margin}$.
3. **Stage 3 — Diversity & Guardrails:**
   - Apply MMR (Maximal Marginal Relevance) to avoid filter bubbles and guarantee merchant diversity.
4. **Serving Infrastructure:**
   - Dockerized FastAPI service deployed on AWS EC2, caching hot user embeddings in Redis.

### 📊 Evaluation & Interview Talking Points
- **Metrics:** Evaluated retrieval with **Recall@300 (94%)** and ranking with **NDCG@10 (0.84)**.
- **Latency Performance:** End-to-end inference executes in **27ms** at p99.
- **Cold Start:** Addressed using content-based metadata fallback and multi-armed bandits ($\epsilon$-greedy).

---

## 🛡️ Project 5 (FinTech / Stripe / AWS): Real-Time Fraud Prevention with Feature Store & Drift Monitoring
> **Big Tech Problem:** How financial institutions and cloud payment platforms (Stripe, PayPal, AWS Financial Services) detect fraud in streaming transactions while preventing model decay.  
> **Target Roles:** MLOps Engineer, Production ML Engineer, Platform Engineer  
> **Key Cloud Stack:** Feast (Feature Store), XGBoost, Evidently AI, MLflow, AWS Kinesis, Redis, Docker, GitHub Actions

### 💡 The Problem
Financial transactions exhibit extreme class imbalance (<0.1% fraudulent). Models suffer from **Training-Serving Skew** when batch SQL queries calculate features differently from real-time API code, and evolving fraud tactics cause silent **Concept Drift**.

### 🏗️ Architecture & Implementation
1. **Unified Feature Store (Feast):**
   - **Offline Store (Amazon S3 / Snowflake):** Produces training datasets using **point-in-time joins** (time-travel) to prevent future data leakage.
   - **Online Store (Redis):** Serves sub-5ms pre-aggregated features (`user_velocity_1h`, `foreign_country_flag`) to the inference API.
2. **Cost-Sensitive Modeling:**
   - Train an **XGBoost Classifier** with Focal Loss, optimized specifically for **PR-AUC (Precision-Recall AUC)** and Recall at fixed False Positive Rate (FPR < 0.5%).
   - Track all experiments and model artifacts in **MLflow Model Registry**.
3. **Automated Drift Detection & Retraining:**
   - Incoming predictions stream to a monitoring worker.
   - **Evidently AI** runs daily statistical checks:
     - **Population Stability Index (PSI)** on transaction amounts and velocity features.
     - **Kolmogorov-Smirnov (KS) Test** on continuous features.
   - If $\text{PSI} \ge 0.25$, an automated **GitHub Actions CI/CD pipeline** triggers model retraining on fresh data, validates against the production Champion model, and deploys a Canary container to AWS.

### 📊 Evaluation & Interview Talking Points
- **Eliminating Skew:** How Feast's single source of truth eliminated feature calculation bugs between data science notebooks and Go/Java production microservices.
- **Imbalance Mastery:** Why Accuracy and ROC-AUC are misleading for fraud, and how PR-AUC protected business revenue.
- **Drift Self-Healing:** The automated alert-to-retraining lifecycle.

---

## ⚡ Project 6 (AI Infrastructure): High-Throughput Enterprise LLM Serving with vLLM & Semantic Caching
> **Big Tech Problem:** How AI cloud infrastructure providers serve thousands of concurrent LLM requests under strict GPU memory limits and sub-second latency budgets.  
> **Target Roles:** AI Infrastructure Engineer, LLM Platform Engineer, Core AI Engineer  
> **Key Cloud Stack:** vLLM (PagedAttention), Llama 3.1 / Mistral, Redis Semantic Cache, OpenTelemetry, Arize Phoenix, Docker, AWS g5.2xlarge

### 💡 The Problem
Enterprise customer-facing applications receive 1,000+ queries per minute. Commercial API costs exceed $20,000/month, and peak p95 latency spikes over 3.5 seconds due to GPU memory fragmentation and KV cache explosion.

### 🏗️ Architecture & Implementation
1. **Optimized Inference Engine (vLLM):**
   - Deploy an open-weight LLM on a cloud GPU instance using **vLLM**.
   - Leverage **PagedAttention** to allocate KV cache memory in non-contiguous physical blocks (similar to OS virtual memory), boosting concurrent batch capacity by 4x.
2. **Two-Tier Latency Reduction:**
   - **Semantic Response Cache (Redis):** Computes lightweight embeddings for incoming questions. If semantic similarity to a recent query exceeds 0.96, the cached response is served in **<15ms** ($0 token cost).
   - **Prompt Caching:** Shares pre-computed KV caches for multi-page static system instructions, cutting Time-To-First-Token (TTFT) by 70%.
3. **Distributed Telemetry & Cost Observability:**
   - Instrument the serving gateway with **OpenTelemetry** exporting traces to **Arize Phoenix**.
   - Monitor real-time operational metrics:
     - **TTFT (Time-to-First-Token)** — Prefill latency.
     - **TPOT (Time-per-Output-Token)** — Decode generation throughput.
     - GPU VRAM utilization and token dollar savings.

### 📊 Evaluation & Interview Talking Points
- **Cost Efficiency:** Slashed monthly LLM infrastructure spend by 72% via semantic caching and vLLM continuous batching.
- **Concurrency:** Scaled single GPU throughput from 8 req/s to 48 req/s without Out-Of-Memory (OOM) errors.
- **Latency SLA:** Reduced p95 latency from 3.2s to 450ms.

---

## 📋 Big Tech & Cloud Alignment Matrix

| Project | Cloud / Big Tech Target | Core Architecture | Key Differentiator |
|---|---|---|---|
| **Serverless IDP Pipeline** | **AWS (Bedrock / Textract)** | EventBridge $\rightarrow$ Lambda $\rightarrow$ Textract $\rightarrow$ Bedrock $\rightarrow$ OpenSearch | Serverless event-driven architecture + automated PII redaction |
| **Multimodal Video Intelligence** | **GCP (Vertex AI / BigQuery)** | GCS $\rightarrow$ Vertex Multimodal Embeddings $\rightarrow$ BigQuery Vector Search $\rightarrow$ Gemini | Sub-100ms vector search across petabytes of video in BigQuery |
| **Autonomous Operations Copilot** | **Google Cloud (Google ADK)** | Google ADK $\rightarrow$ Subagent Delegation $\rightarrow$ FastMCP $\rightarrow$ Cloud Run | Code-first software engineering + `adk web` debugging + MCP tools |
| **Two-Stage Recommendation Engine**| **Meta / Amazon** | Two-Tower Embeddings $\rightarrow$ FAISS HNSW $\rightarrow$ LightGBM LambdaMART | Sub-30ms candidate retrieval + ranking at scale |
| **Real-Time Fraud Pipeline** | **Stripe / FinTech / AWS** | Feast Feature Store $\rightarrow$ XGBoost PR-AUC $\rightarrow$ Evidently PSI Drift CI/CD | Eliminates training-serving skew + automated drift retraining |
| **High-Throughput LLM Serving** | **AI Infra / Scale AI** | vLLM PagedAttention $\rightarrow$ Redis Semantic Cache $\rightarrow$ OpenTelemetry | 4x concurrency gains + TTFT/TPOT latency budget optimization |
