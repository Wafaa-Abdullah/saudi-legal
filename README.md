# Saudi Legal AI API — v3.0

An AI-powered REST API for querying Saudi Arabian legal regulations and systems, built on Retrieval-Augmented Generation (RAG) with multi-model fallback support.

---

## Overview

This API provides natural language access to Saudi legal regulations sourced from the Ministry of Justice. It supports queries in Arabic (formal and colloquial), English, and handles common spelling errors in both languages.

**Performance Metrics**

| Metric | Value |
|---|---|
| Overall Accuracy | 95%+ |
| Law Detection Accuracy | 100% |
| English Query Support | Full |
| Typo Correction Rate | 94% |
| Average Response Time | ~5 seconds |

---

## Deployment — Railway

### Prerequisites
- Railway account at railway.app
- GitHub account
- API keys: Groq, OpenRouter, HuggingFace

### Steps

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/saudi-legal-api.git
git push -u origin main
```

**2. Create Railway Project**
- Go to railway.app
- Select New Project
- Select Deploy from GitHub Repo
- Choose your repository

**3. Set Environment Variables**

Navigate to your Railway project settings and add the following variables:

```
GROQ_API_KEY          =  your_groq_api_key
OPENROUTER_API_KEY    =  your_openrouter_api_key
HF_TOKEN              =  your_huggingface_token
API_SECRET_KEY        =  choose_a_strong_secret_key
HF_REPO_ID            =  WafaaFraih/saudi-legal-moj
CHROMA_PATH           =  ./chroma_db
```

**4. Deploy**

Railway will automatically build and deploy using the provided Dockerfile. First deployment takes approximately 10-15 minutes due to model and dataset loading.

---

## API Reference

### Base URL
```
https://your-app-name.railway.app
```

### Authentication

All endpoints (except `/` and `/health`) require an API key passed via request header:
```
X-API-Key: your_secret_key
```

---

### GET /

Returns basic API information.

**Response**
```json
{
  "name": "Saudi Legal AI",
  "version": "3.0.0",
  "status": "running",
  "docs": "/docs"
}
```

---

### GET /health

Returns current system status. No authentication required.

**Response**
```json
{
  "status": "healthy",
  "chunks": 7410,
  "models": ["Groq llama-3.3", "Groq llama-3.1-8b"],
  "qwen": true,
  "or_models": 1,
  "version": "3.0.0"
}
```

---

### POST /ask

Submit a legal question and receive an answer based on Saudi regulations.

**Headers**
```
X-API-Key: your_secret_key
Content-Type: application/json
```

**Request Body**
```json
{
  "question": "ما هي شروط الزواج؟",
  "include_sources": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| question | string | Yes | Legal question (max 1000 characters) |
| include_sources | boolean | No | Include source references in response. Default: true |

**Response**
```json
{
  "answer": "المرجع: نظام الأحوال الشخصية — المادة الثالثة عشرة\nالإجابة: شروط صحة عقد الزواج هي: تعيين الزوجين، رضا الزوجين، الإيجاب من الولي، شهادة شاهدين.",
  "sources": [
    {
      "law": "نظام الأحوال الشخصية",
      "article": "المادة الثالثة عشرة"
    }
  ],
  "coverage": 100,
  "model": "Groq llama-3.3",
  "duration_ms": 1240,
  "disclaimer": "هذه المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة. يُنصح بمراجعة محامٍ مختص."
}
```

**cURL Example**
```bash
curl -X POST "https://your-app.railway.app/ask" \
  -H "X-API-Key: your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي شروط الزواج؟"}'
```

**Error Responses**

| Status Code | Description |
|---|---|
| 400 | Empty question or question exceeds 1000 characters |
| 401 | Missing or invalid API key |
| 500 | Internal server error |

---

### GET /stats

Returns request statistics and recent activity log. Requires authentication.

**Response**
```json
{
  "total": 150,
  "success": 138,
  "blocked": 10,
  "errors": 2,
  "unique_ips": 12,
  "top_users": [["192.168.1.1", 45], ["10.0.0.2", 30]],
  "recent": [...]
}
```

---

### GET /docs

Interactive Swagger UI documentation. Available at `/docs` on your deployed URL.

---

## Supported Query Types

| Type | Example |
|---|---|
| Formal Arabic | ما هي شروط مزاولة مهنة المحاماة؟ |
| Colloquial Arabic | ايه عقوبة التوثيق بدون رخصه؟ |
| English | What is the penalty for money laundering? |
| Arabic with typos | شروط مزاولت مهنه المحاماه؟ |
| English with typos | penality for money laudering? |
| Practical questions | أنا موظف اشتغلت 3 سنين هل لي مكافأة؟ |
| English practical | I worked 5 years, am I entitled to end of service? |

---

## Available Legal Systems

The API covers 71 legal systems including:

- Ministry of Justice Systems: Notarization, Advocacy, Personal Status, Criminal Procedures, Arbitration, Bankruptcy, Real Estate Registration, Civil Transactions, Anti-Money Laundering, Judiciary
- Labor Law
- Anti-Cybercrime Law
- Personal Data Protection Law
- Companies Law

---

## System Architecture

```
Input Question
    |
    v
Spell Correction  (Arabic and English typo correction)
    |
    v
Translation       (English to Arabic)
    |
    v
Legal Detection   (Determines if question is within scope)
    |
    v
Query Expansion   (Generates multiple query variations)
    |
    v
Hybrid Search     (Semantic search + BM25 + Keyword matching)
    |
    v
Reranking         (Scores and ranks retrieved documents)
    |
    v
Generation        (Groq → Qwen HF → OpenRouter fallback chain)
    |
    v
Post-processing   (Formatting + disclaimer injection)
    |
    v
API Response
```

---

## Model Fallback Chain

| Priority | Provider | Model | Notes |
|---|---|---|---|
| 1 | Groq | llama-3.3-70b-versatile | Primary — fastest |
| 2 | HuggingFace | Qwen2.5-72B-Instruct | Fallback when Groq rate-limited |
| 3 | OpenRouter | Various free models | Last resort |

Rate limiting is handled automatically with per-provider tracking.

---

## Legal Disclaimer

The information provided by this API is intended for reference purposes only and does not constitute legal advice. Users are advised to consult a licensed legal professional for matters requiring formal legal guidance.

---

## Project Structure

```
saudi-legal-api/
    main.py              Application entry point and full pipeline
    requirements.txt     Python dependencies
    Dockerfile           Container configuration
    README.md            This file
    .env.example         Environment variable template
    .gitignore           Git exclusions
    chroma_db/           Persistent vector database (auto-generated)
```
