# Deep Learning Projects

A collection of hands-on AI/ML experiments — a RAG system built from scratch, an AI-powered resume matcher, and an LLM comparison tool. Each project is self-contained and explores a different piece of the modern LLM application stack: retrieval, embeddings, and multi-model evaluation.

## Table of Contents

- [Projects](#projects)
  - [1. Basic RAG](#1-basic-rag)
  - [2. Resume Score Matcher](#2-resume-score-matcher)
  - [3. Model Comparisons](#3-model-comparisons)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

## Projects

### 1. Basic RAG
**Folder:** [`Basic_rag/`](./Basic_rag)

A Retrieval-Augmented Generation pipeline built from first principles — no vector database, no embedding model, just plain Python to understand how RAG actually works under the hood.

**How it works:**
1. Loads a source text document (`Ai.txt`)
2. Splits it into overlapping word chunks
3. Scores each chunk against the user's question using word-overlap (set intersection)
4. Passes the most relevant chunk as context to an LLM (Groq's `llama-3.1-8b-instant`) to generate a grounded answer
5. Returns "not found in document" if no chunk meets the minimum relevance score

**Run it:**
```bash
cd Basic_rag
python Basic_RAG.py
```

**Requires:** `groq` Python package, and a `groq_api_key` environment variable.

---

### 2. Resume Score Matcher
**Folder:** [`Resume_score/`](./Resume_score)

Matches a resume (PDF) against a job description and generates an AI-written compatibility analysis.

**How it works:**
1. Extracts text from a resume PDF using `pdfplumber`
2. Encodes the resume and job description into embeddings with `sentence-transformers`
3. Uses `FAISS` for similarity indexing/search across resume content
4. Sends the retrieved context to an LLM (via the NVIDIA NIM API, `meta/llama-3.3-70b-instruct`) to generate a match analysis

**Run it:**
```bash
cd Resume_score
pip install -r requirements.txt
python Resume_matcher_Score.py
```

**Requires:** `nvidia_api_key` environment variable, plus a resume PDF path configured in the script.

---

### 3. Model Comparisons
**Folder:** [`models comparisions/`](./models%20comparisions)

Benchmarks multiple LLMs (served via Groq) on the same prompt to compare response quality and latency side by side.

**Models compared:**
- GPT-OSS-120B (`openai/gpt-oss-120b`)
- Llama 3.3 70B (`llama-3.3-70b-versatile`)
- Qwen 3 32B (`qwen/qwen3-32b`)

**How it works:**
1. Loads a dataset and generates a statistical summary (`pandas.describe()`)
2. Sends the same analysis prompt to all three models
3. Times each model's response
4. Plots a bar chart comparing response times using `matplotlib`

**Run it:**
```bash
cd "models comparisions"
python models.py
```

**Requires:** `groq` Python package, `groq_api_key` environment variable, and a dataset CSV path configured in the script.

---

## Tech Stack

| Category | Tools |
|---|---|
| LLM Providers | Groq, NVIDIA NIM (OpenAI-compatible API) |
| Embeddings & Search | Sentence-Transformers, FAISS |
| PDF Processing | pdfplumber |
| Data & Visualization | pandas, matplotlib |
| Language | Python 3 |

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Kiran07222/deep_learning_projects.git
   cd deep_learning_projects
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies for the project you want to run (see each project's section above; `Resume_score/requirements.txt` covers that project's full environment).
4. Set the required API keys as environment variables (e.g. `groq_api_key`, `nvidia_api_key`) before running any script.

## Repository Structure

```
deep_learning_projects/
├── Basic_rag/
│   ├── Basic_RAG.py
│   └── Ai.txt
├── Resume_score/
│   ├── Resume_matcher_Score.py
│   └── requirements.txt
├── models comparisions/
│   ├── models.py
│   └── images/
└── README.md
```

## Author

**Vuyyalawada Kiran (Nani)**
B.Tech CSE, Vasireddy Venkatadri Institute of Technology (VVIT)

---

*This is a personal learning repository documenting hands-on exploration of RAG systems, embeddings, and LLM evaluation.*
