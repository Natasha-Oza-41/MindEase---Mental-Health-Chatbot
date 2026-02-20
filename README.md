# Mental Health Chatbot — AI-Powered Support System

> A production-style mental health support chatbot built with FastAPI, RAG, and a
> responsible AI safety layer. Designed for portfolio demonstration and interview readiness.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Why LLM Over Rule-Based Systems](#3-why-llm-over-rule-based-systems)
4. [Architecture](#4-architecture)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Justification](#6-design-decisions--justification)
7. [Safety Mechanisms](#7-safety-mechanisms)
8. [Hallucination Mitigation Strategy](#8-hallucination-mitigation-strategy)
9. [Limitations](#9-limitations)
10. [Future Improvements](#10-future-improvements)
11. [Evaluation Approach](#11-evaluation-approach)
12. [How to Scale This System](#12-how-to-scale-this-system)
13. [Cost Optimization Ideas](#13-cost-optimization-ideas)
14. [Setup Instructions](#14-setup-instructions)
15. [API Reference](#15-api-reference)
16. [Project Structure](#16-project-structure)
17. [How to Explain This Project in a 10-Minute Interview](#17-how-to-explain-this-project-in-a-10-minute-interview)

---

## 1. Project Overview

This project is a locally-runnable, AI-powered mental health support chatbot that:

- Responds empathetically to users discussing mental health concerns
- Uses **Retrieval-Augmented Generation (RAG)** to ground responses in verified mental health content, reducing hallucination
- Maintains **per-session conversation memory** for coherent multi-turn dialogue
- Has a **crisis detection safety layer** that intercepts high-risk messages before they reach the LLM, returning pre-written, clinically appropriate crisis responses with real helpline resources
- Logs all conversations to JSONL files for auditability
- Exposes a clean REST API via **FastAPI**

**Tech stack:**
| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| LLM | OpenAI API (gpt-4o-mini) |
| Embeddings | sentence-transformers (local) |
| Vector Store | ChromaDB (local, persistent) |
| Conversation Memory | In-memory (Python dict) |
| Logging | Python logging + JSONL |
| Testing | pytest |

---

## 2. Problem Statement

Mental health support is critically under-resourced globally. According to the WHO:
- Over **280 million people** suffer from depression worldwide.
- There is a global shortage of mental health professionals (treatment gaps exceed 70% in low-income countries).
- Most people wait **years** before seeking help — largely due to stigma and lack of accessible support.

**The opportunity:** AI chatbots cannot replace therapists, but they can provide:
- 24/7 accessible, judgment-free, anonymous support
- Psychoeducation and evidence-based coping strategies
- A bridge to professional help for those who are hesitant to seek it

**The challenge:** Deploying AI in mental health requires responsible design — especially around crisis situations where the wrong response could cause harm.

---

## 3. Why LLM Over Rule-Based Systems

Traditional rule-based mental health chatbots (like Woebot's early versions) used decision trees and scripted responses.

| Dimension | Rule-Based | LLM-Based |
|-----------|-----------|-----------|
| Language understanding | Pattern matching, rigid | Contextual, nuanced |
| Response variety | Repetitive, feels scripted | Natural, varied |
| Handling novel phrasing | Fails on unseen inputs | Generalises well |
| Personalisation | Minimal | Adapts to user's language and context |
| Maintenance | Must manually update every path | Prompt engineering is more flexible |
| Safety | Predictable but brittle | More flexible but needs guardrails |

**The hybrid approach used here:** LLM for language generation + deterministic rule-based crisis detection for safety. This gives us the fluency of LLMs with the reliability of rules where it matters most.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                   │
│  POST /api/chat                                                   │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────┐     ┌─────────────────┐                        │
│  │   ChatService │────▶│ Crisis Detector │ ◀── Keyword taxonomy  │
│  │ (Orchestrator)│     └────────┬────────┘                        │
│  └──────┬───────┘              │ CRITICAL/HIGH                   │
│         │                      ▼                                  │
│         │               Pre-written safe response                 │
│         │               + real crisis resources                   │
│         │                                                          │
│         ▼ (if no override)                                        │
│  ┌──────────────┐                                                 │
│  │  RAG Pipeline │                                                │
│  │  ┌─────────┐  │  user query embedding                         │
│  │  │ChromaDB │◀─┤──────────────────────                         │
│  │  │ (local) │  │                                                │
│  │  └────┬────┘  │  top-K relevant chunks                        │
│  └───────┼───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Prompt Builder                                        │        │
│  │  system_prompt + RAG_context + conversation_history   │        │
│  └──────────────────────────┬───────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│                    ┌──────────────────┐                           │
│                    │   OpenAI LLM     │  (gpt-4o-mini)            │
│                    └────────┬─────────┘                           │
│                              │                                     │
│                              ▼                                     │
│                    ┌──────────────────┐                           │
│                    │  Response + optional safety disclaimer        │
│                    └────────┬─────────┘                           │
│                              │                                     │
│          ┌───────────────────┼──────────────────┐                │
│          ▼                   ▼                   ▼                │
│   ConversationMemory   JSONL Log File      API Response           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `ChatService` | `app/services/chat_service.py` | Orchestrates the full request pipeline |
| `CrisisDetector` | `app/core/crisis_detector.py` | Keyword-based severity classification |
| `RAGPipeline` | `app/core/rag_pipeline.py` | Document embedding, storage, retrieval |
| `ConversationMemory` | `app/core/memory.py` | Per-session sliding window history |
| `OpenAIClient` | `app/core/llm_client.py` | LLM API abstraction |
| `logger` | `app/utils/logger.py` | App logs + conversation JSONL logs |

---

## 5. Data Flow

```
User Input
    │
    │  "I've been feeling really anxious and can't sleep."
    │
    ▼
[Crisis Detector]
    │  → Scan for crisis keywords across 4 severity tiers
    │  → Severity: LOW  (keyword: "anxious")
    │  → requires_override() → False
    │
    ▼
[RAG Pipeline]
    │  → Embed user query with sentence-transformers (local)
    │  → Query ChromaDB for top-3 most relevant chunks
    │  → Returns: chunks from anxiety_management.md + stress_coping.md
    │
    ▼
[Prompt Builder]
    │  → system_prompt (persona + boundaries)
    │  + RAG context (retrieved chunks)
    │  + conversation history (last N turns from memory)
    │  + current user message
    │
    ▼
[LLM — gpt-4o-mini]
    │  → Generates an empathetic, grounded response
    │
    ▼
[Safety Disclaimer]
    │  → Appended because severity = LOW
    │
    ▼
[Memory + Log]
    │  → Add turn to ConversationMemory[session_id]
    │  → Append to conversations_YYYYMMDD.jsonl
    │
    ▼
API Response → { session_id, response, crisis_info, sources }
```

---

## 6. Design Decisions & Justification

### Decision 1: No LangChain
**Choice:** Build RAG from scratch using ChromaDB + sentence-transformers directly.

**Why:** LangChain is an abstraction that hides the mechanics. Building it from scratch demonstrates understanding of each step — embedding, storage, retrieval, prompt injection — which is what interviewers want to see. LangChain also adds significant complexity and version instability for a project of this size.

### Decision 2: Local Embeddings (sentence-transformers)
**Choice:** `all-MiniLM-L6-v2` runs on CPU with no API key.

**Why:** Zero cost per query. Works offline after the initial model download (~80MB). For the scale of a portfolio project, this is entirely sufficient. Production upgrade: swap to OpenAI embeddings (`text-embedding-3-small`) for higher quality.

### Decision 3: ChromaDB for Vector Store
**Choice:** ChromaDB with persistent local storage.

**Why:** No server to run. Data persists across restarts. Simple Python API. Production upgrade path is clear (swap to Pinecone, Weaviate, or pgvector).

### Decision 4: In-Memory Conversation Store
**Choice:** Python `defaultdict` keyed by `session_id`.

**Why:** Simplest solution that works correctly for a single-instance local deployment. The interface (`get_memory()`, `add_message()`, `get_history()`) is designed so the implementation can be swapped for Redis without changing the rest of the codebase.

### Decision 5: Deterministic Crisis Detection First
**Choice:** Keyword-based crisis detection always runs before the LLM.

**Why:** LLMs are non-deterministic. For life-safety situations, you cannot rely on probabilistic outputs. Rule-based detection is fast, predictable, auditable, and cannot be "jailbroken." The cost of false positives (showing crisis resources unnecessarily) is far lower than false negatives (missing a real crisis).

### Decision 6: Sliding Window Memory
**Choice:** Keep only the last `MAX_CONVERSATION_HISTORY * 2` messages.

**Why:** LLM context windows have token limits. Keeping all history would eventually overflow the context. A sliding window keeps the conversation coherent while preventing token overflow.

### Decision 7: JSONL Conversation Logs
**Choice:** One JSON line per conversation turn, stored in dated files.

**Why:** JSONL is append-only (no locking issues), easily queryable with standard tools, and each line is a valid JSON object. It can be directly imported into analytics tools or used for future fine-tuning datasets.

---

## 7. Safety Mechanisms

### Layer 1: Pre-LLM Crisis Gate
- Runs on every message before any LLM call
- 4-tier severity classification (CRITICAL → HIGH → MEDIUM → LOW → NONE)
- CRITICAL and HIGH severity: LLM is **not called** — a pre-written, clinically appropriate response with real emergency resources is returned immediately
- This gate cannot be circumvented by the LLM's behaviour

### Layer 2: System Prompt Guardrails
The system prompt explicitly instructs the LLM to:
- Never diagnose mental health conditions
- Never prescribe or recommend specific medications
- Always recommend professional help for serious concerns
- Identify itself as an AI, not a therapist
- Prioritise directing users to emergency services if in danger

### Layer 3: Safety Disclaimer
For LOW and MEDIUM severity signals, a disclaimer is appended to the LLM's response reminding the user that the bot is an AI and providing crisis text line info.

### Layer 4: LLM Fallback
If the LLM call fails for any reason (API error, timeout, rate limit), the fallback response explicitly provides crisis line information — the system fails safe.

### Layer 5: Conversation Logging
All conversations are logged with session ID, timestamp, and crisis severity level — creating an audit trail that could be reviewed for safety incidents.

---

## 8. Hallucination Mitigation Strategy

LLMs can generate plausible-sounding but incorrect information. In mental health, this is especially dangerous (e.g., wrong dosage, incorrect diagnosis criteria, invalid coping advice).

**Strategies used in this system:**

1. **RAG grounding:** Responses are anchored to curated, verified mental health documents. The system prompt instructs the LLM to use the provided context. This significantly reduces the LLM's reliance on its internal (potentially outdated or incorrect) knowledge.

2. **Strict system prompt boundaries:** The LLM is told not to diagnose or prescribe, which prevents the most dangerous class of hallucinations.

3. **Scope limitation:** The LLM's `max_tokens` is capped at 500, keeping responses concise and reducing the surface area for hallucination.

4. **Temperature:** Set to 0.7 (moderate creativity). Lower values (0.1-0.3) produce more factual but robotic responses; higher values increase creativity but also hallucination risk.

5. **Source attribution:** Retrieved document sources are returned in the API response, allowing a human reviewer to verify the origin of information.

**What this does NOT fully solve:** The LLM can still hallucinate information not covered by the RAG documents, or misapply retrieved context. This is why the system includes clear disclaimers that it is not a licensed therapist and professional help should be sought.

---

## 9. Limitations

Be honest about these in interviews — it demonstrates maturity.

| Limitation | Description |
|-----------|-------------|
| **Not a clinical tool** | Cannot assess or treat mental health conditions. Should never be used as a replacement for professional care. |
| **Crisis detection is shallow** | Keyword matching produces false positives (e.g., "I want to kill this project") and misses indirect expressions of distress. A proper clinical system would use a fine-tuned classifier. |
| **No persistent memory across restarts** | Conversation history is lost when the server restarts. Requires a database for persistence. |
| **Single-language** | Only handles English. Mental health support is especially critical in under-resourced language communities. |
| **No identity verification** | The same `session_id` could be used by different people, or the same person could lose their session ID. |
| **No human escalation** | In production, a detected crisis should trigger a workflow to notify a human supervisor, not just surface a phone number. |
| **LLM consistency** | Non-deterministic outputs mean the same question may get different answers. |
| **RAG coverage gaps** | The system only knows what is in the indexed documents. Queries outside this scope may result in less accurate responses. |
| **No user authentication** | Anyone can call the API with any session ID. |

---

## 10. Future Improvements

| Improvement | Priority | Effort |
|------------|---------|--------|
| Replace keyword crisis detection with a fine-tuned classifier (e.g., fine-tuned BERT on suicide risk datasets) | High | High |
| Persistent conversation storage (PostgreSQL or Redis) | High | Medium |
| Human-in-the-loop escalation for high-severity crises | High | High |
| User authentication (JWT tokens) | Medium | Medium |
| Sentiment analysis as an additional crisis signal | Medium | Medium |
| Multi-language support (translate input → process → translate output) | Medium | High |
| Evaluation dashboard (track average crisis severity, response quality) | Medium | Medium |
| Swap to OpenAI embeddings for higher retrieval quality | Low | Low |
| Add a simple web UI (React or plain HTML/JS) | Low | Medium |
| Fine-tune a smaller model on mental health conversation datasets | Low | High |
| Integrate with therapist booking systems | Future | High |

---

## 11. Evaluation Approach

Evaluating a mental health chatbot is non-trivial. Unlike a search engine, there is no single "correct" answer.

### Automated Metrics
| Metric | Tool/Method | Limitation |
|--------|-------------|-----------|
| **RAG Retrieval Quality** | NDCG / Recall@K using manually labelled query-document pairs | Requires creating an evaluation dataset |
| **Response Relevance** | Cosine similarity between query and response embeddings | Doesn't capture empathy or clinical accuracy |
| **Crisis Detection F1** | Precision/Recall on labelled crisis messages | Requires a labelled dataset |
| **LLM-as-Judge** | Use GPT-4 to score responses on empathy, accuracy, safety (1-5 scale) | Meta-evaluation; still requires careful prompt engineering |

### Human Evaluation
The most reliable evaluation method. Have evaluators (ideally with mental health background) rate responses on:
- **Empathy:** Does it feel heard and validated?
- **Accuracy:** Is the information factually correct?
- **Safety:** Does it appropriately refer to professional help?
- **Naturalness:** Does it sound human and conversational?
- **Helpfulness:** Would this actually help someone feeling this way?

### Safety Evaluation (Red-Teaming)
Deliberately probe the system with:
- Direct crisis statements (verify override works)
- Indirect crisis expressions (test detection gaps)
- Attempts to get the bot to diagnose/prescribe
- Jailbreaking attempts ("pretend you're a doctor")
- Ambiguous statements that could be crisis or not

### A/B Testing (Production)
Compare different system prompt versions, RAG configurations, or models on real (anonymised) conversations using outcome metrics like session length, user satisfaction, and escalation rate.

---

## 12. How to Scale This System

The current design is intentionally simple. Here is the upgrade path for production scale:

### Stateless API Tier (Easy)
- Deploy FastAPI on any container platform (Docker + Kubernetes, AWS ECS, Railway)
- Multiple API instances can run in parallel — they are stateless since all state is external

### Conversation Memory (Required for scale)
- Replace the in-memory dict with **Redis** (low latency, TTL support) or **PostgreSQL**
- Use `session_id` as the key
- This allows any API instance to serve any session

### Vector Store (For large document collections)
- Swap ChromaDB for **Pinecone**, **Weaviate**, or **pgvector** (PostgreSQL extension)
- These support horizontal scaling and have managed hosted options

### LLM
- Use **OpenAI's batch API** for non-real-time processing (cheaper)
- Consider **caching** common queries (semantic caching with a vector store)
- Explore open-source alternatives (Llama 3, Mistral) self-hosted on GPU servers to reduce per-token cost

### Safety at Scale
- Upgrade crisis detection to a dedicated ML classifier running as a microservice
- Add a human review queue for flagged conversations
- Implement rate limiting per session to prevent abuse

### Observability
- Add distributed tracing (OpenTelemetry)
- Track per-request latency, LLM token usage, cache hit rates
- Set up alerts for elevated crisis detection rates

---

## 13. Cost Optimization Ideas

Running this system at scale on OpenAI's API can be expensive. Strategies to reduce cost:

| Strategy | Potential Saving | Trade-off |
|---------|----------------|-----------|
| **gpt-4o-mini** over gpt-4o | ~20x cheaper per token | Slightly lower response quality |
| **Limit `max_tokens`** (currently 500) | Reduces output cost | May truncate long responses |
| **Semantic caching** — return cached response for semantically similar queries | 30-70% reduction for repetitive questions | Complexity; requires cache invalidation |
| **Local/open-source LLM** (Ollama + Llama 3 8B) | Near-zero per-token cost after GPU setup | Requires GPU hardware; lower quality |
| **Local embeddings** (already implemented) | Free retrieval step | Slightly lower quality than OpenAI embeddings |
| **Shorter system prompt** | Fewer input tokens per request | Reduces LLM guardrail effectiveness |
| **Batch non-real-time requests** | Up to 50% discount via OpenAI Batch API | Adds latency (results in hours, not seconds) |
| **RAG to reduce LLM tokens** | Context compression reduces prompt length | None significant |

---

## 14. Setup Instructions

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key (get one at https://platform.openai.com/)
- ~500MB free disk space (for model download + vector store)

### Step 1: Clone / Download the Project
```bash
# If you have git
git clone <your-repo-url>
cd mental-health-chatbot

# Or just navigate to the project folder
cd "AI Powered - Mental Health chatbot"
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80MB) the first time it runs. This is a one-time download.

### Step 4: Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...your-key-here...
```

### Step 5: Ingest Documents into the Vector Store
```bash
python scripts/ingest_documents.py
```
You should see output like:
```
✓  Successfully indexed 87 document chunks.
   You can now start the server:  uvicorn app.main:app --reload
```

### Step 6: Start the Server
```bash
uvicorn app.main:app --reload
```
The server will start at `http://localhost:8000`

### Step 7: Verify It's Working
Open your browser and go to:
- `http://localhost:8000/docs` — Interactive Swagger UI
- `http://localhost:8000/api/health` — Health check

### Step 8: Send Your First Message
Using the Swagger UI at `/docs`, or via curl:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-test-session",
    "message": "I have been feeling really anxious about work lately."
  }'
```

### Running Tests
```bash
pytest tests/ -v
```

---

## 15. API Reference

### POST `/api/chat`
Send a message to the chatbot.

**Request body:**
```json
{
  "session_id": "unique-user-session-id",
  "message": "I've been feeling anxious and overwhelmed."
}
```

**Response:**
```json
{
  "session_id": "unique-user-session-id",
  "response": "I hear you — feeling anxious and overwhelmed is really tough...",
  "crisis_info": null,
  "sources": ["anxiety_management.md", "stress_coping.md"]
}
```

**Crisis response example:**
```json
{
  "session_id": "unique-user-session-id",
  "response": "I'm very concerned about what you've shared...",
  "crisis_info": {
    "severity": "critical",
    "detected_keywords": ["suicide"],
    "resources": [
      {
        "name": "988 Suicide & Crisis Lifeline",
        "contact": "Call or text 988 (US)",
        "available": "24/7",
        "url": "https://988lifeline.org"
      }
    ]
  },
  "sources": null
}
```

### DELETE `/api/chat/{session_id}`
Clear all conversation history for a session.

### GET `/api/health`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "rag_status": "ready",
  "active_sessions": 3
}
```

---

## 16. Project Structure

```
mental-health-chatbot/
│
├── app/
│   ├── main.py                    # FastAPI app, lifespan, middleware, routes
│   ├── config.py                  # Pydantic settings (reads from .env)
│   │
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py            # POST /api/chat, DELETE /api/chat/{id}
│   │       └── health.py          # GET /api/health
│   │
│   ├── core/
│   │   ├── llm_client.py          # Abstract LLM base + OpenAI implementation
│   │   ├── memory.py              # In-memory session conversation store
│   │   ├── crisis_detector.py     # Keyword-based crisis detection
│   │   └── rag_pipeline.py        # ChromaDB + sentence-transformers RAG
│   │
│   ├── services/
│   │   └── chat_service.py        # Orchestration: crisis → RAG → LLM → log
│   │
│   └── utils/
│       └── logger.py              # App logger + JSONL conversation logger
│
├── data/
│   └── mental_health_docs/        # Source documents for RAG
│       ├── anxiety_management.md
│       ├── depression_support.md
│       ├── stress_coping.md
│       ├── mindfulness_techniques.md
│       └── crisis_resources.md
│
├── scripts/
│   └── ingest_documents.py        # One-time setup: embed docs into ChromaDB
│
├── tests/
│   ├── test_crisis_detector.py    # Unit tests for crisis detection
│   └── test_chat_service.py       # Unit tests for chat orchestration (mocked)
│
├── logs/                          # App logs + conversation JSONL (auto-created)
├── vector_store/                  # ChromaDB persisted data (auto-created)
│
├── .env.example                   # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 17. How to Explain This Project in a 10-Minute Interview

This section will help you communicate this project clearly, confidently, and technically. Practice this structure until it feels natural.

---

### Structured Explanation Flow (the "story arc")

**Minute 0-1: Hook with the problem**
> "I built an AI-powered mental health support chatbot. The core motivation is that mental health is severely under-resourced globally — most people can't or won't seek professional help due to cost, stigma, or availability. An AI chatbot doesn't replace a therapist, but it can provide 24/7 accessible psychoeducation and a non-judgmental space. The challenge is doing this responsibly."

**Minute 1-2: The responsible AI angle (this is what makes it memorable)**
> "The most interesting engineering challenge wasn't the chatbot itself — it was the safety layer. I built a crisis detection system that intercepts every message *before* it reaches the LLM. If someone expresses suicidal ideation, the LLM is never called. Instead, a pre-written, clinically appropriate response with real emergency resources is returned. This is intentional — LLMs are probabilistic. For life-safety situations, you can't rely on a model to always do the right thing."

**Minute 2-4: Architecture walkthrough**
> "The system has five main components: [walk through the architecture diagram] — the API layer, the crisis detector, the RAG pipeline, the LLM client, and the conversation memory. The ChatService orchestrates all of them. Let me explain the flow..."
> Walk through the data flow from the diagram above.

**Minute 4-6: RAG deep dive (technical highlight)**
> "RAG stands for Retrieval-Augmented Generation. Instead of relying purely on the LLM's training data — which could be outdated or hallucinated — I created a local knowledge base of curated mental health documents covering anxiety, depression, stress, mindfulness, and crisis resources. When a user sends a message, I embed it using a local sentence-transformer model, retrieve the three most semantically similar document chunks from ChromaDB, and inject them into the LLM's context window. This grounds the response in verified content and significantly reduces hallucination."

**Minute 6-7: Technical decisions**
> "I chose not to use LangChain. I wanted to understand each component of RAG deeply — the embedding step, the vector similarity search, the context injection — so I built it from scratch. I used ChromaDB because it persists to disk with no server setup needed. For embeddings, I used all-MiniLM-L6-v2, which runs locally on CPU and costs nothing per query."

**Minute 7-8: Limitations and honest reflection**
> "I'm aware of the limitations. Keyword-based crisis detection has false positives — 'I want to kill this bug in my code' would trigger it. A proper system would use a fine-tuned classifier. The conversation memory is in-RAM, so it doesn't persist across restarts — a production system would use Redis. And the system should have a human-in-the-loop escalation path for crisis situations, not just a phone number."

**Minute 8-10: Handle follow-up questions (see below)**

---

### Key Technical Points to Highlight

1. **Safety-first design:** Crisis detection runs before LLM, and always will.
2. **RAG from scratch:** Not just "I used LangChain" — you understand each step.
3. **Responsible AI thinking:** You've considered hallucination, false positives, failure modes.
4. **Clean architecture:** Dependency injection, abstract base classes, singleton pattern, separation of concerns.
5. **Testability:** Mocked dependencies in unit tests — shows production mindset.
6. **Logging:** JSONL conversation logs for auditability — shows you think about production operations.

---

### Likely Follow-Up Questions & Strong Answers

---

**Q: "What is RAG and why did you use it?"**

> "RAG stands for Retrieval-Augmented Generation. LLMs have two problems in specialised domains: their training data might not include the latest or most accurate information, and they can hallucinate — generating confident but wrong answers. RAG solves this by maintaining a separate knowledge base. When a query arrives, we embed it, retrieve the most relevant document chunks using vector similarity search, and inject them into the LLM's context. The LLM then generates a response grounded in that retrieved content. In mental health specifically, giving wrong coping advice or wrong information about medications could cause real harm, so grounding responses in verified documents is essential."

---

**Q: "How does your crisis detection work? Isn't keyword matching too simple?"**

> "You're right — keyword matching is intentionally simple, and that's a feature, not a bug, at this layer. The advantage is that it is deterministic, auditable, fast, and cannot be bypassed by an adversarial input that tricks the LLM. The limitation is false positives, like flagging 'I want to kill this bug in my code', and false negatives for indirect expressions of distress. A production system would layer this with a fine-tuned classifier — something like a BERT model fine-tuned on suicide risk datasets like the CLPsych shared task datasets. But even then, I'd keep the keyword check as a first pass because it's so fast and reliable for the clearest cases. The key design principle is: fail safe. I'd rather show crisis resources to someone who doesn't need them than miss someone who does."

---

**Q: "Why not use LangChain?"**

> "I deliberately chose not to use LangChain for this project. LangChain is a useful framework, but it abstracts away the mechanics of RAG — the embedding step, the vector query, the context injection. Since one goal of this project was to demonstrate that I understand how RAG works at a fundamental level, building it with the raw components made more sense. I'm comfortable with LangChain and would use it in a production codebase where speed of development matters more. But for showing technical depth in an interview context, building it from scratch is more valuable."

---

**Q: "How would you handle a real user in crisis using this system?"**

> "Honestly? This system alone is not sufficient for real crisis intervention. What I've built shows the technical pattern, but production-ready crisis support would require: a clinical team reviewing the crisis detection rules and response scripts, a human escalation pathway when high-severity signals are detected (trigger a notification to a human supervisor, not just a phone number), integration with real crisis services, and regular safety audits. The system as built should be clearly labelled as a support tool, not a crisis intervention system. The 988 Lifeline and Crisis Text Line are the right resources — my system's job is to surface them quickly and clearly."

---

**Q: "How would you scale this to thousands of users?"**

> "The current bottleneck at scale is the in-memory conversation store — it would run out of RAM and not work across multiple API instances. I'd replace it with Redis, which is fast, supports TTL for session expiry, and works across any number of API nodes. The FastAPI app itself is stateless and can be containerised with Docker and scaled horizontally behind a load balancer. ChromaDB would be replaced with a managed vector database like Pinecone or pgvector. For the LLM, I'd implement semantic caching to avoid calling the API for frequently-repeated queries, and move to GPT-4o-mini if not already to reduce per-token cost. At very high scale, I'd consider fine-tuning a smaller open-source model on mental health conversations to eliminate per-token API costs entirely."

---

**Q: "How do you evaluate whether the chatbot is actually helpful?"**

> "This is genuinely hard. You can't rely purely on automated metrics because the most important qualities — empathy, safety, and psychological validity — require human judgment. My approach would be three-tiered: First, automated metrics for the retrieval layer — precision@K for whether the right documents are retrieved. Second, LLM-as-judge evaluation where GPT-4 scores responses on empathy, accuracy, and safety using a rubric. Third, and most important, human evaluation by raters with a mental health background. I'd also do systematic red-teaming — deliberately testing with crisis inputs, jailbreaking attempts, and edge cases — to verify safety properties. Ultimately, a clinical psychologist should sign off on any version intended for real users."

---

**Q: "What would you do differently if you were building this for production?"**

> "Several things. Crisis detection would use a fine-tuned ML classifier, not just keywords. I'd add user authentication so sessions are tied to verified identities. I'd build a human-in-the-loop escalation workflow — flagged conversations would trigger a notification to a human supervisor within minutes. I'd add a rate limiter to prevent abuse. Conversation data would be stored encrypted in a database with proper data retention policies and GDPR compliance. And I'd never deploy without a clinical review by licensed mental health professionals. The system as built is a technical prototype that demonstrates the architecture — production deployment in mental health requires much more careful validation."

---

**Q: "Why gpt-4o-mini and not a larger model?"**

> "Cost and latency. gpt-4o-mini is about 20x cheaper per token than gpt-4o, and for a support chatbot where responses are capped at 500 tokens, the quality difference is acceptable. The RAG grounding compensates for some of the capability gap — by providing relevant context, even a smaller model can give high-quality, accurate responses. For the crisis detection and safety-critical responses, I'm not using the LLM at all — those use pre-written scripts — so model capability doesn't matter there. If I found that users were consistently getting poor responses for complex queries, I'd upgrade the model or add a routing layer that escalates complex cases to a more powerful model."

---

### How to Defend Architectural Decisions

When challenged on any decision, use this structure:

1. **Acknowledge the trade-off honestly** — "You're right, this has limitations"
2. **Explain the reasoning** — "I chose this because..."
3. **State the upgrade path** — "In production, I would..."

This shows engineering maturity. Interviewers are not looking for perfection — they are looking for evidence that you understand trade-offs and can reason about them.

---

*Built with FastAPI · ChromaDB · sentence-transformers · OpenAI API*
*Designed for responsible AI deployment in mental health contexts*
