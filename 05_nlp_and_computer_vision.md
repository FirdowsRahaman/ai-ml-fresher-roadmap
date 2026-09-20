# Module 05: NLP & Computer Vision for Freshers

This module covers the basics of Natural Language Processing (NLP) and Computer Vision — two of the most common application areas in AI/ML engineering.

---

## Section 1: Natural Language Processing (NLP) Basics

### Q1: What is NLP (Natural Language Processing)?

**Answer:**

NLP is the branch of AI that helps computers **understand, interpret, and generate human language** (text and speech).

**Real-World Examples:**

- Google Translate (language translation)

- ChatGPT (text generation and conversation)

- Gmail's Smart Reply (suggesting short replies)

- Siri / Alexa (understanding voice commands)

- Amazon reviews sentiment analysis (positive/negative/neutral)

---

### Q2: What is a Token? What is Tokenization?

**Answer:**

A **Token** is the smallest meaningful unit that a language model processes. A token is roughly equivalent to a word or a piece of a word.

**Tokenization** is the process of breaking raw text into tokens before feeding it into an ML model.

```

Sentence: "I love machine learning!"

Word Tokenization:    ["I", "love", "machine", "learning", "!"]

Subword Tokenization:  ["I", "love", "mach", "##ine", "learn", "##ing", "!"]

Character Tokenization: ["I", " ", "l", "o", "v", "e", ...]

```

**Why tokens, not words?**

- "unhappiness" → ["un", "happi", "ness"] — Subword tokens handle rare or complex words without needing a vocabulary of millions of words.

- Most modern LLMs (GPT, BERT, LLaMA) use **subword tokenization** (BPE or WordPiece).

```python

# Example using HuggingFace tokenizer

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer.tokenize("Machine learning is fascinating!")

print(tokens) # ['machine', 'learning', 'is', 'fascinating', '!']

print(f"Number of tokens: {len(tokens)}") # 5

```

---

Q2b: In General — What ARE Tokens for Inputs and Outputs?

**Answer:**

A **token** is primarily a unit used to represent text for a language model. A tokenizer breaks text into tokens before the model processes it.

A token can be:

A complete word

Part of a word (a **subword**)

Punctuation

Another small piece of text

For example:

"Machine learning is fascinating!"

Possible tokens:
["Machine", "learning", "is", "fascinating", "!"]

But a word can also be split into subwords:

"unhappiness"
   ↓
["un", "happi", "ness"]

This is called **subword tokenization**.

 What happens when we send different types of input?

Different modalities are processed differently.

TEXT
"What is in this image?"
    ↓
  Text Tokenizer
    ↓
  Text Tokens


IMAGE
[photo of a cat]
    ↓
Vision Encoder / Image Processor
    ↓
Image patches / visual representations

The multimodal model can then combine these representations.

So:

**Text → tokenizer → text tokens**

**Image → vision encoder → visual representations**

**Audio → audio encoder → audio representations**

**Video → video encoder → video representations**

Do **not** assume that every modality is converted into ordinary text tokens.

 What about the output?

The output depends on what the model is generating.

For a text-generation model:

Model
 ↓
Output text tokens
 ↓
"Artificial Intelligence is..."

The model generates text tokens, typically one token at a time.

This is called **autoregressive generation**.

But if the model generates an image:

Model
 ↓
Image-generation process
 ↓
Image representation
 ↓
 Generated image

The final output is an image, not a sequence of text tokens.

Similarly, a speech-generation system can produce audio, and a video-generation system can produce video.

Some multimodal models may internally use terms such as **image tokens**, **audio tokens**, or **visual tokens**. These are model-specific representations and should not be confused with text subword tokens.

 Simple Mental Model

         ┌── Text tokenizer ──→ Text tokens
         │
INPUT ───────────┼── Vision encoder ──→ Visual representation
         │
         ├── Audio encoder ──→ Audio representation
         │
         └── Video encoder ──→ Video representation
                   ↓
                Multimodal Model
                   ↓
             ┌────────────┼────────────┐
             ↓      ↓      ↓
          Text output  Image output Audio/Video
          (text tokens) (image)   (generated media)

**Key idea:**

**Token is not a universal synonym for every kind of AI data.**

**Text is commonly represented using text tokens. Other modalities use model-specific representations, which may sometimes also be called tokens.**

 Why Tokens Matter in Practice

For text-based LLM APIs, usage and billing are commonly measured using **input tokens** and **output tokens**.

For example:

Your prompt:
"Summarize this document..."
    ↓
Input text tokens

Model generates:
"The document explains..."
    ↓
Output text tokens

Providers can also have model-specific usage or pricing rules for images, audio, and video.

There is no universal rule such as:

"One image = 500 tokens"

"10 seconds of audio = 500 tokens"

"5 seconds of video = 5,000 tokens"

The exact accounting depends on the model, provider, modality, resolution, sampling, and architecture.

**Interview tip:**

If asked whether an image is "a token," say:

**"An image is not a text token. A multimodal model may convert the image into patches, features, or visual tokens internally, depending on its architecture."**
---

#### Input Tokens — What the model READS

| What you send in | Token Type | Example |

|---|---|---|

| Text (question, prompt) | Text Tokens | "What is AI?" → ~4 tokens |

| Image | Image Patch Tokens | 512×512 photo → ~680 tokens |

| Audio clip | Audio Frame Tokens | 10 seconds of speech → ~500 tokens |

| Video | Video Frame Tokens | 10-second video → ~5,000+ tokens |

| Code | Code Tokens (same as text) | 100 lines of Python → ~700 tokens |

```

You type: "Describe this image → [photo attached]"

      ↓             ↓

    Text Tokens      Image Patch Tokens

    (\~5 tokens)        (\~500 tokens)

      ↓

   Total Input = \~505 tokens fed into the model

```

---

#### Output Tokens — What the model GENERATES

| What the model outputs | Token Type | Example |

|---|---|---|

| Text reply (GPT, Gemini) | Text Tokens | 200-word answer → ~270 tokens |

| Generated image (DALL-E) | Image Patch Tokens | 1024×1024 image decoded from patch tokens |

| Speech (ElevenLabs) | Audio Tokens | 5 seconds of speech → ~250 audio tokens |

| Generated video (Sora) | Video Tokens | 5-second clip decoded from video tokens |

| Code (GitHub Copilot) | Code Tokens | A function decoded as text tokens |

```

Model generates: "Artificial Intelligence is the simulation..."

           ↓

   Output Text Tokens — produced ONE BY ONE, left to right

   "Artificial" → "Intel" → "ligence" → "is" → ...

   (Called AUTOREGRESSIVE generation — one token at a time)

```

---

#### Universal Mental Model (Works for ALL Modalities)

```

┌──────────────────────────────────────────────────────────────┐

│           ANY AI MODEL              │

│                               │

│  INPUT (Any type)       OUTPUT (Any type)      │

│  ──────────────────      ──────────────────      │

│  Text  → Tokens ──┐     ┌── Tokens → Text       │

│  Image → Tokens ──┤     ├── Tokens → Image      │

│  Audio → Tokens ──┤ MODEL  ├── Tokens → Audio      │

│  Video → Tokens ──┘ ─────► └── Tokens → Video      │

│                               │

│ Everything becomes tokens IN. Everything comes out as tokens│

└──────────────────────────────────────────────────────────────┘

```

---

#### Why Tokens Matter in Practice (API Billing)

When you call any LLM API (OpenAI, Gemini, Claude), **you pay per token** — input AND output separately:

```

Example API call to GPT-4o:

Your prompt:  "Summarize this document..." → 1,500 input tokens

Attached image: [chart photo]        →  500 input tokens

Model's reply: "The document covers..."   →  300 output tokens

                     ────────────────────

Total billed:                 2,300 tokens

Cost (approximate):

Input : 2,000 tokens × $5/million = $0.010

Output:  300 tokens × $15/million = $0.005

Total :               \~$0.015 per call

```

**Key rules for interviews:**

- Output tokens are **3–5× more expensive** than input tokens.

- The **context window** = the maximum (input + output) tokens the model handles in one go.

- RAG exists partly because you can't fit a 500-page document into the context window — so you only retrieve the relevant 3–5 chunks (a few hundred tokens) instead!

---

### Q3: What is Embedding? What is a Word Embedding?

**Answer:**

An **Embedding** is a numerical **vector representation** of data—such as text, images, audio, or other content—designed to capture useful relationships such as similarity.

- Words with similar meaning get similar vector representations.

- The computer can then do math on these vectors.

```

"King"  → [0.8, 0.2, -0.5, 0.9, ...]

"Queen" → [0.75, 0.3, -0.4, 0.85, ...]  ← Similar to King!

"Apple" → [-0.3, 0.9, 0.1, -0.5, ...]  ← Very different from King/Queen

```

**Famous example:** `King - Man + Woman ≈ Queen`

**Popular embedding models:**

- Word2Vec (classic)

- GloVe (classic)

- Sentence-BERT / text-embedding-ada-002 (modern, for entire sentences)

---

### Q3b: If Embeddings exist for Text — What about Images, Audio, and Video?

**Answer:**

Embeddings are NOT just for text! **Any type of data can be converted into a vector (embedding)**. This is the core idea that makes modern multimodal AI possible.

The concept is always the same regardless of the data type:

> **Raw data** (pixels / sound waves / video frames) → **Encoder Model** → **Vector of numbers** (Embedding)

Similar items → similar vectors. Different items → different vectors.

---

#### Image Embeddings

An image encoder (like a CNN or Vision Transformer) looks at the entire image and compresses it into a single vector that captures what is in the image.

```

[Photo of a Cat] → Image Encoder → [0.82, -0.3, 0.55, 0.91, ...] (512 numbers)

[Photo of a Dog] → Image Encoder → [0.79, -0.2, 0.51, 0.88, ...] (512 numbers) ← Close to cat!

[Photo of a Car] → Image Encoder → [-0.4, 0.8, -0.2, -0.6, ...] (512 numbers) ← Far from cat/dog

```

**Popular image embedding models:**

- **CLIP (OpenAI)** — Creates embeddings for BOTH images and text in the same shared space! This allows you to search images using text descriptions.

- **ResNet, EfficientNet** — CNN-based image encoders.

**Use Cases:** Image similarity search, reverse image search (like Google Lens), visual product recommendations.

---

#### Audio Embeddings

Audio is first converted into a visual representation (spectrogram), then an encoder compresses it into a vector capturing the sound's meaning, tone, or speaker identity.

```

[Audio: "Hello, how are you?"] → Audio Encoder → [0.3, 0.7, -0.1, ...]

[Audio: "Hello, how are you?"] → Audio Encoder → [0.31, 0.68, -0.12, ...] ← Same phrase, same speaker → similar!

[Audio: Dog barking]      → Audio Encoder → [-0.5, 0.2, 0.9, ...]  ← Very different!

```

**Popular audio embedding models:**

- **Whisper (OpenAI)** — Primarily a speech-to-text / speech-recognition model; it should not be described as a general-purpose audio embedding model.

- **wav2vec 2.0 (Meta)** — Raw audio → embeddings for speech recognition.

- **CLAP** — Audio-language contrastive model (like CLIP but for audio+text).

**Use Cases:** Music recommendation (Spotify), speaker recognition, sound classification.

---

#### Video Embeddings

Video is a sequence of image frames over time. Video encoders process both the **visual content** AND **temporal changes** (how scenes evolve) to produce an embedding.

```

[Video: Person waving hello] → Video Encoder → [0.5, 0.3, 0.8, -0.2, ...]

[Video: Person waving bye]  → Video Encoder → [0.48, 0.31, 0.77, -0.18, ...] ← Similar gesture!

[Video: Car crash]      → Video Encoder → [-0.3, 0.9, -0.5, 0.6, ...]  ← Very different!

```

**Popular video embedding models:**

- **VideoMAE, TimeSFormer** — For video understanding.

- **Video-language and multimodal embedding models** — can produce representations useful for video understanding or retrieval; exact capabilities depend on the model and API.

**Use Cases:** YouTube video recommendations, video content moderation, sports analytics.

---

#### The Big Picture — Multimodal Embeddings (Everything in One Space!)

The most important modern development is that some multimodal embedding models, such as **CLIP**, are trained to align representations from different modalities in a shared vector space.

```

Text: "a dog playing fetch" → [0.7, 0.3, -0.2, 0.8, ...]

Image: [Photo of dog + ball] → [0.69, 0.31, -0.21, 0.79, ...] ← Very close! 

Text: "a red sports car"  → [0.1, -0.5, 0.9, 0.2, ...]

Image: [Photo of red Ferrari] → [0.11, -0.51, 0.88, 0.21, ...] ← Very close! 

```

**This enables:**

- **Text-to-Image search** — Type "sunset over mountains" → Find all matching photos in your database.

-  **Image-to-Text search** — Upload a photo of a shoe → Find similar products on an e-commerce site.

-  **Audio-to-Text search** — Hum a tune → Find the matching song.

#### Summary Table: Embeddings Across All Modalities

| Data Type | Input | Encoder Type | Embedding Use Case |

|---|---|---|---|

| **Text** | Words / sentences | BERT, GPT, Word2Vec | Semantic search, chatbots, RAG |

| **Image** | Pixels (H × W × 3) | CNN, ViT, CLIP | Image search, face recognition |

| **Audio** | Sound waveform | Wav2Vec, Whisper | Music recommendation, speech recognition |

| **Video** | Frames over time | VideoMAE, Gemini | Video recommendation, action detection |

| **Code** | Code text | CodeBERT, GitHub Copilot | Code search, bug detection |

| **Multimodal** | Text + Image + Audio | CLIP, Gemini, GPT-4o | Cross-modal search, visual Q&A |

---

### Q4: What is Sentiment Analysis?

**Answer:**

Sentiment Analysis classifies text as **Positive, Negative, or Neutral** to understand the emotional tone.

**Use Cases:**

- Analyzing product reviews on Amazon/Flipkart.

- Monitoring brand reputation on Twitter/X.

- Customer satisfaction scoring in support tickets.

```python

from transformers import pipeline

# Pre-built sentiment analysis pipeline

sentiment = pipeline("sentiment-analysis")

result = sentiment("I absolutely loved this product, it works perfectly!")

print(result) # [{'label': 'POSITIVE', 'score': 0.9998}]

result2 = sentiment("The delivery was delayed and the product was broken.")

print(result2) # [{'label': 'NEGATIVE', 'score': 0.9993}]

```

---

### Q5: What is Named Entity Recognition (NER)?

**Answer:**

NER identifies and classifies **named entities** (real-world objects) in text such as people, organizations, locations, dates, and monetary values.

**Example:**

```

Input: "Apple Inc. was founded by Steve Jobs in Cupertino in 1976."

NER Output:

Apple Inc. → ORGANIZATION

Steve Jobs → PERSON

Cupertino  → LOCATION

1976    → DATE

```

---

## Section 2: Computer Vision Basics

### Q6: What is Computer Vision?

**Answer:**

Computer Vision is the branch of AI that enables computers to **understand and interpret images and videos** — like how humans use their eyes and brain together.

**Real-World Examples:**

- Face recognition (iPhone Face ID)

- Detecting objects in autonomous vehicles (pedestrians, traffic lights, cars)

- Medical image analysis (detecting tumors in X-rays)

- Quality control in manufacturing (detecting defects on products)

---

### Q7: What is a CNN (Convolutional Neural Network)?

**Answer:**

A CNN is a type of neural network specially designed for processing images. Instead of looking at an entire image at once, it scans the image with small filters to detect local patterns like edges, corners, textures, and shapes.

```

Input Image

 ↓

[Convolutional Layer] ← Detects edges and basic patterns

 ↓

[Pooling Layer]    ← Reduces size, keeps important info

 ↓

[Convolutional Layer] ← Detects more complex patterns

 ↓

[Fully Connected]   ← Makes the final classification

 ↓

Output: "Cat" / "Dog" / ...

```

**Popular CNN architectures:** VGG16, ResNet, MobileNet, EfficientNet.

---

### Q8: What is Transfer Learning? Why is it useful for freshers and small teams?

**Answer:**

Transfer Learning means taking a model already pre-trained on a large dataset and **reusing it** for your own smaller task, instead of training from scratch.

**Analogy:** You already learned to ride a bicycle → learning a motorcycle is much easier because many skills transfer over.

**In practice:**

1. Take a ResNet50 model trained on ImageNet (1.4 million images, 1000 classes).

2. Replace the final classification layer with your own (e.g., 2 classes: cat vs dog).

3. Train only the new final layer on your small dataset (even 500 images can work well!).

```python

from torchvision import models

import torch.nn as nn

# Load pretrained model

model = models.resnet50(pretrained=True)

# Freeze all layers (don't update these weights)

for param in model.parameters():

param.requires\_grad = False

# Replace the final layer for your task (2 classes)

model.fc = nn.Linear(model.fc.in_features, 2)

# Only the final layer will be trained

```

**Why useful for freshers?**

- You don't need massive data or computing power.

- State-of-the-art results with minimal code.

---

### Q9: What is the difference between Image Classification, Object Detection, and Image Segmentation?

**Answer:**

| Task | What it does | Output | Example |

|---|---|---|---|

| **Image Classification** | Labels the entire image with one class | "Cat" | Is this photo a cat or dog? |

| **Object Detection** | Finds where objects are + classifies them | Bounding boxes + labels | Draws a box around each car in a traffic camera |

| **Image Segmentation** | Labels every single pixel of the image | Pixel-level mask | Outlines exact shape of road, sky, buildings for a self-driving car |

---

## Section 3: Key NLP Concepts

### Q10: What is the Bag of Words (BoW) model?

**Answer:**

Bag of Words is one of the simplest ways to convert text into numbers for ML. It creates a vector where each number represents the **count of how many times each word appears** in the document.

```

Vocabulary: ["good", "bad", "movie", "great", "acting"]

Sentence 1: "good movie great acting" → [1, 0, 1, 1, 1]

Sentence 2: "bad movie bad acting"   → [0, 2, 1, 0, 1]

```

**Limitation:** BoW ignores word order and meaning. "I like dogs, not cats" and "I like cats, not dogs" would produce the same vector!

---

### Q11: What is TF-IDF?

**Answer:**

TF-IDF (Term Frequency-Inverse Document Frequency) is an improvement over BoW. It gives **higher scores to words that appear frequently in a specific document but rarely in other documents** — highlighting words that are uniquely important to that document.

- Common words like "the", "is", "a" appear in all documents → get low TF-IDF score.

- Specific words like "machine", "gradient", "neural" in an ML article → get high TF-IDF score.

```python

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [

"machine learning is great",

"deep learning is a subset of machine learning",

"I enjoy learning new things"

]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())

```