# Module 06: Generative AI, LLMs & RAG — Fresher Edition

This is the most exciting and in-demand area in AI today. Every fresher interviewing for an AI/ML role is now expected to know the basics of Generative AI, LLMs, Tokens, Prompting, and RAG.

---

## Section 1: What is Generative AI?

### Q1: What is Generative AI?
**Answer:**
Generative AI is a type of AI that can **create new content** — text, images, audio, code, or video — that did not exist before. It learns patterns from massive amounts of existing data and uses that knowledge to generate something new.

**Examples of Generative AI in daily use:**
- **ChatGPT** → Generates human-like text conversations
- **DALL-E / Midjourney** → Generates images from text descriptions
- **GitHub Copilot** → Generates code suggestions
- **Suno / Udio** → Generates music
- **Synthesia** → Generates AI avatar videos

**Contrast with Traditional AI:**
- Traditional/Discriminative AI: *"Is this email spam or not?"* → Classifies existing data.
- Generative AI: *"Write me a professional email declining this job offer."* → Creates new content.

---

### Q2: What is the difference between Discriminative AI and Generative AI?
**Answer:**

| | Discriminative AI | Generative AI |
|---|---|---|
| **Task** | Classifies or predicts from existing data | Creates new data |
| **Question it answers** | "What category does this belong to?" | "What would new data in this domain look like?" |
| **Examples** | Spam detector, Face recognizer, Credit scoring | ChatGPT, DALL-E, Copilot, Gemini |
| **Output** | A label, number, or probability | New text, image, audio, or code |

---

### Q3: What is an LLM (Large Language Model)?
**Answer:**
A **Large Language Model (LLM)** is a type of AI model trained on enormous amounts of text data (books, websites, code, research papers — hundreds of billions of words) to understand and generate human language.

**The word "Large" refers to:**
1. **Massive training data** (trillions of tokens from the internet)
2. **Billions of parameters** (learned weights — GPT-4 has ~1.8 trillion)
3. **Huge compute required** (thousands of GPUs running for weeks/months)

**Popular LLMs:**
| Model | Company | Open / Closed |
|---|---|---|
| GPT-4 / GPT-4o | OpenAI | Closed (API access) |
| Gemini 1.5 Pro | Google DeepMind | Closed (API access) |
| Claude 3.5 | Anthropic | Closed (API access) |
| LLaMA 3 | Meta AI | Open Source |
| Mistral / Mixtral | Mistral AI | Open Source |

---

### Q4: What is a Token in the context of LLMs?
**Answer:**
In LLMs, a **token** is the basic unit of text that the model reads and generates. Think of tokens as chunks of text — roughly 3-4 characters each, or about ¾ of a word on average.

```
Sentence: "Artificial intelligence is transforming every industry."

Tokens:  ["Art", "ificial", " intelligence", " is", " transform", "ing", " every", " industry", "."]
Count:   9 tokens (approximately)
```

**Why does it matter?**
- LLMs have a **context window** (maximum token limit they can read at once).
 - GPT-3.5: 4,096 tokens (~3,000 words)
 - GPT-4 Turbo: 128,000 tokens (~96,000 words)
 - Gemini 1.5 Pro: 1,000,000 tokens!
- **You pay for API calls by tokens** (both input + output tokens count).

---

### Q4b: If a Token is for Text — What is the Equivalent for Images, Video, and Audio?
**Answer:**
This is one of the most important questions in modern AI! "Token" is originally a text concept, but modern multimodal AI models process all kinds of data. Here is how each modality is broken down into processable units:

---

#### 🖼️ Images → **Patches** (Image Tokens / Visual Tokens)
Images are split into small fixed-size squares called **patches** — typically 16×16 or 32×32 pixels each. Each patch is then converted into a vector (like a token embedding) and fed into the model.

```
Full Image (384 × 384 pixels)
    ↓ Split into 16×16 patches
[Patch1][Patch2][Patch3] ... [PatchN]  → 576 visual tokens
  ↓      ↓        ↓
 Vector 1   Vector 2    Vector N  (each becomes a "token" for the model)
```

**Used In:**
- **ViT (Vision Transformer)** — Splits images into 16×16 patches and processes them like text tokens.
- **GPT-4o, Gemini, Claude** — Convert image patches to visual tokens, then process text and image tokens together.

**1 image ≈ 300–1000 tokens** (depending on resolution and model). That's why images are expensive in multimodal API calls!

---

#### 🎥 Video → **Frame Patches** (Video Tokens)
Video is a sequence of images (frames). Models process videos by:
1. **Sampling frames** — e.g., 1 frame per second or every 2 seconds.
2. **Splitting each frame into patches** — same as image tokenization.
3. **Adding time information** — so the model knows which frame came first.

```
Video (10 seconds @ 1 fps = 10 frames)
  Frame 1 → 576 patches
  Frame 2 → 576 patches
  ...
  Frame 10 → 576 patches
  Total ≈ 5,760 video tokens (very expensive!)
```

**Used In:** Gemini 1.5 Pro (can process up to 1 hour of video), Google VideoPoet.

---

#### 🔊 Audio → **Mel Spectrogram Frames** (Audio Tokens)
Audio cannot be processed as raw sound waves directly. It is first converted into a **Mel Spectrogram** — a visual representation of how sound frequencies change over time (essentially turning audio into an image). Then those spectrogram chunks become tokens.

```
Audio Waveform (speech)
    ↓
Convert to Mel Spectrogram (frequency vs time image)
    ↓
Split into time windows (e.g., 20ms chunks)
    ↓
Each chunk → Audio Token Vector → Fed to Transformer
```

**Typical rate:** ~50 audio tokens per second of speech.

**Used In:**
- **Whisper (OpenAI)** — Speech-to-text model, processes 30-second audio chunks.
- **Gemini** — Can natively process audio input.
- **ElevenLabs / Suno** — Generate speech or music using audio token generation.

---

#### Summary Table: "Tokens" Across All Modalities

| Modality | Basic Unit | Actual Name | Typical Size |
|---|---|---|---|
| **Text** | Chunk of text (~¾ word) | Token | ~4 characters |
| **Image** | Small square region of pixels | Patch / Visual Token | 16×16 or 32×32 pixels |
| **Video** | Image patch from one frame at one time | Video Token | 16×16 pixels × time step |
| **Audio** | Short time window of sound frequencies | Audio Token / Frame | ~20 milliseconds |
| **Code** | Same as text (code is text!) | Token | ~4 characters |

> **Key Insight for Interviews:** Modern multimodal models like **GPT-4o** and **Gemini** all convert different inputs (text, image, audio) into a **common representation (vectors/embeddings)** in the same shared "space" so the Transformer can process them all together in one unified model!

---

### Q5: What is a Prompt? What is Prompt Engineering?
**Answer:**
A **Prompt** is the text instruction you give to an LLM to get a response. The quality of the prompt dramatically affects the quality of the output.

**Prompt Engineering** is the practice of writing effective prompts to get the best possible outputs from LLMs.

#### Types of Prompting:

**Zero-Shot Prompting** — No examples, just a direct instruction:
```
Prompt: "Classify this review as Positive or Negative: 'The food was awful.'"
LLM:  "Negative"
```

**Few-Shot Prompting** — Provide a few examples before the actual task:
```
Prompt:
 Review: "The service was excellent!" → Positive
 Review: "Worst experience ever."   → Negative
 Review: "I will never come back."  → ?

LLM: "Negative"
```

**Chain-of-Thought (CoT) Prompting** — Ask the model to reason step by step:
```
Prompt: "Q: John has 5 apples. He gives 2 to Mary and buys 3 more. How many does he have?
     Think step by step."
LLM:  "Step 1: John starts with 5. Step 2: Gives 2 → 5-2=3. Step 3: Buys 3 → 3+3=6. Answer: 6"
```

---

### Q6: What is the difference between a Foundation Model and a Fine-Tuned Model?
**Answer:**
- **Foundation Model:** A large, general-purpose model pre-trained on broad internet data. Can do many tasks but isn't specialized at any single one.
 - *Example:* LLaMA 3 (can write poetry, summarize news, answer questions, generate code).
- **Fine-Tuned Model:** A foundation model that has been **further trained on a specific domain or task** with additional labeled data, making it much better at that specific task.
 - *Example:* LLaMA 3 + Medical research papers → Better at answering clinical questions.
 - *Example:* GPT-4 + customer service conversations → Better chatbot for your company.

---

### Q7: What is Hallucination in LLMs? Why does it happen?
**Answer:**
**Hallucination** is when an LLM confidently generates text that is **factually incorrect or completely made up** — but presents it as if it is true.

**Examples:**
- LLM invents a fake research paper and gives it a plausible-sounding author, journal, and DOI.
- LLM says a celebrity died in a year they didn't.
- LLM invents a court case or legal precedent.

**Why it happens:**
- LLMs are trained to predict the next most likely token, not to be factually accurate.
- They don't know what they don't know — they generate plausible-sounding text even outside their knowledge.
- If the answer isn't in their training data, they "fill in the gap" with a confident guess.

**Mitigation:**
- Use RAG (Retrieval-Augmented Generation) to ground answers in real documents.
- Ask the model to say "I don't know" when uncertain.
- Always verify LLM outputs for critical tasks.

---

## Section 2: RAG — Retrieval-Augmented Generation

### Q8: What is RAG (Retrieval-Augmented Generation)?
**Answer:**
RAG is a technique that **combines a search/retrieval system with an LLM** to give the model access to specific, up-to-date, or private information that wasn't in its training data.

**The Problem RAG Solves:**
- LLMs have a **knowledge cutoff** (e.g., trained up to April 2024 — knows nothing after).
- LLMs don't know your **company's private documents, policies, or real-time data**.
- LLMs hallucinate when they don't have knowledge.

**How RAG Works (Step by Step):**
```
1. PREPARATION (Done once):
  Your Documents → Split into chunks → Convert to Embeddings → Store in Vector Database

2. USER ASKS A QUESTION:
  User: "What is our company's refund policy?"
     ↓
3. RETRIEVE relevant chunks from Vector DB (similar to user's question)
     ↓
4. INJECT retrieved chunks into the LLM's prompt as context:
  Prompt = "Based on the following context: [POLICY TEXT HERE]
       Answer the user's question: 'What is the refund policy?'"
     ↓
5. LLM GENERATES a grounded, accurate answer
```

---

### Q9: What is a Vector Database? Why do we need it for RAG?
**Answer:**
A **Vector Database** is a special type of database that stores data as **embedding vectors** (lists of numbers representing meaning) and is optimized for finding the most **semantically similar** entries to a query.

**Why regular databases (SQL) don't work:**
- SQL searches for exact keyword matches: `WHERE text LIKE '%refund%'`
- If the document says "money back guarantee" instead of "refund," SQL finds nothing.
- A Vector DB understands that "refund," "money back guarantee," and "return policy" all mean the same thing!

**Popular Vector Databases:**
- **FAISS** (Free, by Meta, runs locally)
- **Chroma** (Free, popular for local RAG development)
- **Pinecone** (Cloud-based, scalable)
- **Weaviate, Qdrant, Milvus** (Open source alternatives)

---

### Q10: What is Fine-Tuning vs RAG? When to use which?
**Answer:**

| | Fine-Tuning | RAG |
|---|---|---|
| **What it does** | Updates the model's weights with new training data | Adds external knowledge at query-time without changing the model |
| **Data needed** | Lots of labeled examples | Just your documents (unstructured) |
| **When to use** | Teaching the model a new style, tone, or specialized skill | Giving the model access to specific facts or private knowledge |
| **Cost** | High (GPU training required) | Low (just store documents in a vector DB) |
| **Knowledge updates** | Requires full retraining | Just add/update documents in the vector DB |
| **Example** | Training a customer service chatbot to always respond politely in your brand voice | Answering questions about a specific product manual |

> **Fresher interview tip:** Most real-world GenAI applications use **RAG** for knowledge, not fine-tuning, because it's cheaper, faster to update, and easier to implement.

---

## Section 3: Key Concepts & Terminology

### Q11: What is Context Window? Why does it matter?
**Answer:**
The **Context Window** is the maximum amount of text (measured in tokens) that an LLM can process in a single request — both your input (prompt) AND the model's output combined.

**Why it matters:**
- If your document is 50,000 words but the model's context window is only 4,096 tokens (~3,000 words), the model can't read the entire document at once.
- This is why RAG exists — instead of feeding the whole 50,000-word document, you retrieve only the 3-5 most relevant paragraphs (few hundred tokens).

```
Context Window Examples:
 GPT-3.5-Turbo:  4,096 tokens (~3,000 words)
 GPT-4o:     128,000 tokens (~96,000 words)
 Gemini 1.5 Pro: 1,000,000 tokens (~750,000 words)!
```

---

### Q12: What are Parameters in an LLM? Why do we say "70B model"?
**Answer:**
**Parameters** are the learnable numerical values (weights) inside a neural network. More parameters generally means a more capable model that can learn more complex patterns.

- "70B model" means a model with **70 Billion parameters**.
- Each parameter is a single number (float). A 70B model in 32-bit precision requires **280 GB of VRAM** just to load!
- In 4-bit quantization (Q4), the same model needs only **~40 GB**.

```
Common Model Sizes:
 7B parameters → Runs on consumer GPU (RTX 3090/4090 with 24GB VRAM)
 13B parameters → Needs ~2 GPUs or high-end workstation
 70B parameters → Needs professional GPU cluster or quantization
 1.8T parameters → GPT-4 (requires massive data centers)
```

---

### Q13: What is Quantization in the context of LLMs?
**Answer:**
Quantization reduces the precision (number of bits) used to store each parameter, drastically shrinking the model's memory footprint so it can run on consumer hardware.

| Precision | Bits per parameter | 7B Model Size | Quality |
|---|---|---|---|
| FP32 (Full) | 32 bits | ~28 GB | Best |
| FP16 / BF16 | 16 bits | ~14 GB | Near-identical |
| INT8 (Q8) | 8 bits | ~7 GB | Very good |
| INT4 (Q4) | 4 bits | ~3.5 GB | Good (slight quality loss) |

**Use Case:** Running LLaMA 3 (7B) locally on your laptop with 8GB RAM using Q4 quantization via tools like **Ollama** or **LM Studio**.
