# ⚖️ Saudi Legal AI Assistant (RAG Pipeline)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square&logo=chainlink)](#)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-F15A24?style=flat-square)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=000)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#)

A production-ready, highly accurate **Retrieval-Augmented Generation (RAG)** system designed to provide expert-level legal consultations based on Saudi Arabian law. This system ingests, processes, and retrieves information from thousands of official legal articles, utilizing an advanced LLM pipeline to deliver precise, context-grounded answers in natural Arabic.

---

## 🎯 Key Features

* **Advanced Semantic Routing:** Implemented a robust keyword and intent detection system to route queries to the correct legal domain (e.g., Labor Law, Criminal Procedures, Personal Status).
* **Colloquial Arabic Processing:** Engineered a custom normalization dictionary to translate Egyptian/Saudi slang (e.g., "اتفصلت", "عاوز اطلق") into formal Arabic (Fusha) prior to vector search, significantly increasing retrieval accuracy.
* **Hybrid Retrieval System:** Combines dense vector search (`ChromaDB` + `paraphrase-multilingual-MiniLM-L12-v2`) with sparse keyword matching (`BM25`) to ensure no critical legal article is missed.
* **Custom OCR JSON Parser:** Developed a specialized fallback parser using Regex to cleanly extract text from malformed or corrupted JSON datasets generated via OCR.
* **Zero Hallucination Guardrails:** Strict Prompt Engineering and "Out of Scope" filters ensure the LLM only answers based on retrieved context, gracefully refusing non-legal queries.
* **LLM Fallback Mechanism:** Utilizes `Gemini 2.5 Flash` as the primary reasoning engine, with an automated, latency-aware fallback to `Llama-3.3-70B` (via Groq) to ensure 100% uptime.
* **Responsive UI:** Features a mobile-friendly, clean HTML/CSS/JS frontend served directly via FastAPI.

---

## 🧠 System Architecture

1.  **Data Ingestion & Parsing:**
    * Ingests 8,000+ articles from Hugging Face datasets and local JSON files.
    * Custom Regex parsing handles corrupted OCR data and extracts `law_name` and `article_number` metadata.
2.  **Chunking & Embedding:**
    * Documents are processed using `RecursiveCharacterTextSplitter` (chunk size = 1500) to prevent truncation of lengthy legal articles.
    * Embeddings generated via Sentence-Transformers and stored in ChromaDB.
3.  **The Pipeline (`/ask` endpoint):**
    * User Query -> Normalization (Slang to Fusha) -> Intent Routing -> Query Expansion -> Hybrid Search (BM25 + Vector) -> Custom Reranking (prioritizing primary laws over executive regulations) -> LLM Generation -> Post-processing (Clean UI Source Tags).

---

## 🚀 Tech Stack

* **Backend Framework:** FastAPI, Uvicorn, Python 3.11
* **AI/RAG Framework:** LangChain, Hugging Face `datasets`
* **Vector Database & Search:** ChromaDB, `rank_bm25`
* **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **LLMs:** Google Gemini API (`gemini-2.5-flash`), Groq API (`llama-3.3-70b-versatile`)
* **Deployment:** Docker, Hugging Face Spaces

---

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Saudi-Legal-AI-Assistant.git](https://github.com/YourUsername/Saudi-Legal-AI-Assistant.git)
    cd Saudi-Legal-AI-Assistant
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set Environment Variables:**
    Create a `.env` file and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_key
    GEMINI_API_KEY=your_gemini_key
    HF_TOKEN=your_huggingface_token
    HF_REPO_ID=WafaaFraih/saudi-legal-moj
    ```
4.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 7860
    ```
5.  **Access the UI:**
    Open your browser and navigate to `http://localhost:7860`.

---
*Developed by [Wafaa Abdullah Yassin Fraih](https://github.com/Wafaa-Abdullah) as a specialized Applied AI engineering project.*
