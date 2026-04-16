---
title: ScamGuard Live
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

<div align="center">

# 🛡️ ScamGuard v5.0 — Neural Defense Grid

### AI-Powered Real-Time Vishing Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

<p align="center">
  <strong>A cinematic, production-grade vishing (voice phishing) detection engine</strong><br>
  <em>Combining Multilingual MiniLM embeddings + Logistic Regression + tiered keyword heuristics<br>with a next-gen holographic security dashboard</em>
</p>

---

</div>

## ✨ Features

### 🧠 Hybrid AI Engine (v5.0)
- **Multilingual MiniLM Sentence Embeddings** (384-dim) for semantic understanding
- **Logistic Regression** classifier with 93%+ accuracy
- **Tiered Keyword Heuristic Engine** — India-specific scam detection (OTP, Aadhaar, KYC, UPI, etc.)
- **Temporal Threat Scorer** — cumulative threat tracking with exponential time-decay
- **Text Preprocessing Pipeline** — URL/phone/email masking, whitespace normalization
- **LRU Embedding Cache** (512 entries) for repeated text optimization
- **Explainable AI (XAI)** — human-readable threat explanations per analysis
- **Hinglish/Multilingual Support** — Hindi, English, and mixed-language detection

### 🎬 Cinematic Dashboard ("Neural Defense Grid")
- **3D Holographic Environment** — Three.js neural network with dual rotating icosahedrons
- **DNA Helix Particles** — threat-reactive color shifting (cyan → amber → red)
- **Liquid Gel Threat Meter** — GSAP physics-based animations
- **Keyword Highlighting** — detected keywords color-coded by severity tier
- **Skeleton Loading States** — smooth UX during analysis
- **Analysis History** — localStorage persistence with replay capability
- **Chromatic Aberration** — RGB split overlay intensifies with threat level
- **Scanline Effect** — CRT-inspired retro overlay
- **Spatial Audio** — dynamic ambient hum, stereo panning, heartbeat LFO on DROP
- **Jarvis Voice Mode** — "Hey ScamGuard" wake word + voice commands
- **KONAMI Code Easter Egg** — ↑↑↓↓←→←→BA unlocks confusion matrix dev mode
- **5 Color Themes** — Cyan, Emerald, Sentinel, Amber, Phantom
- **Command Palette** — Ctrl+K searchable command launcher
- **Boot Screen** — auto-skip or manual skip
- **Mobile Ready** — device orientation parallax, responsive layout

### 🔐 Security Architecture (v5.0)
- **Configurable CORS** — no more wildcard `*` in production
- **Input Validation** — Pydantic models with HTML stripping, length limits
- **Security Headers** — X-Content-Type-Options, X-Frame-Options, Referrer-Policy
- **CSP Meta Tag** — Content Security Policy for XSS prevention
- **WebSocket Heartbeat** — auto-timeout for idle connections
- **Session Auto-Purge** — stale sessions cleaned every 5 minutes
- **Secure Temp Files** — `tempfile.mkstemp()` for audio uploads
- **Request ID Tracking** — every request gets a unique ID for debugging

### 📊 Production Backend (v5.0)
- **API Versioning** — all endpoints under `/api/v1/` with legacy backward compatibility
- **Structured Logging** — `logging` module with timestamps, levels, session IDs
- **Async NLP Inference** — `asyncio.to_thread()` avoids blocking the event loop
- **Batch Analysis** — analyze up to 50 texts in a single request
- **Metrics Endpoint** — total analyses, latency stats, active sessions
- **User Feedback** — submit corrections for future model improvement
- **Deep Health Check** — model inference verification, uptime, version info
- **Global Exception Handler** — proper JSON error responses, never crashes
- **FastAPI Lifespan** — modern startup/shutdown (no deprecated `@on_event`)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Frontend (index.html)                   │
│  Three.js · GSAP · Web Speech API · Web Audio API         │
│  localStorage History · Keyword Highlighting · Skeleton   │
│  ──────────────────────────────────────────────────────── │
│  WebSocket Client ←→ Real-time analysis stream             │
└──────────────────────┬───────────────────────────────────┘
                       │ WebSocket (ws://host/ws/{session})
┌──────────────────────▼───────────────────────────────────┐
│             Backend (FastAPI + Uvicorn v5.0)               │
│  main.py  ─────→  scamguard_enhanced.py                   │
│  ├─ Session Store        ├─ classify_intent()             │
│  ├─ WebSocket Handler     ├─ TemporalThreatScorer         │
│  ├─ Threshold Control     ├─ Multilingual-MiniLM (384d)   │
│  ├─ Metrics Collector     ├─ Logistic Regression Model    │
│  ├─ Rate Limiting         ├─ Embedding Cache (LRU)        │
│  ├─ Input Validation      ├─ Text Preprocessing           │
│  └─ Session Auto-Purge   └─ Explainable AI (XAI)          │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ScamGuard.git
cd ScamGuard

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python train_vishing_model.py

# Start the server
python main.py
# Or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to `http://localhost:8000`

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `PORT` | `8000` | Server port |
| `ENV` | `development` | `development` or `production` |
| `MAX_TEXT_LENGTH` | `5000` | Max characters per analysis |
| `SESSION_MAX_AGE` | `3600` | Session TTL in seconds |
| `ENCODER_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Transformer model |

### Docker

```bash
docker build -t scamguard .
docker run -p 7860:7860 scamguard
```

### Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Upload all project files
3. The `Dockerfile` handles everything automatically

---

## 🔌 API Documentation

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Single text analysis |
| `POST` | `/api/v1/analyze-batch` | Batch analysis (up to 50 texts) |
| `POST` | `/api/v1/analyze-audio` | Audio file → transcribe → analyze |
| `POST` | `/api/v1/reset/{session_id}` | Reset session scorer |
| `POST` | `/api/v1/feedback` | Submit prediction feedback |
| `GET` | `/api/v1/health` | Deep health check |
| `GET` | `/api/v1/keywords` | Keyword tier dictionary |
| `GET` | `/api/v1/metrics` | System metrics |
| `GET` | `/api/v1/version` | Build & version info |

### WebSocket

```
ws://host/ws/{session_id}
```

**Send:**
```json
{ "type": "chunk", "text": "Your text here" }
{ "type": "reset" }
{ "type": "set_thresholds", "alert": 0.55, "drop": 0.80 }
{ "type": "ping" }
```

**Receive:**
```json
{
  "type": "analysis",
  "is_scam": true,
  "nlp_confidence": 87.5,
  "keyword_weight": 6,
  "matched_keywords": [{"word": "otp", "weight": 3, "tier": "CRITICAL"}],
  "cumulative_ct": 0.742,
  "ct_percent": 74.2,
  "status": "ALERT",
  "explanation": "Highly deceptive semantic patterns detected. Crucial security markers requested (OTP).",
  "latency_ms": 45.2
}
```

### Example: cURL

```bash
# Single analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your account is blocked. Share your OTP immediately."}'

# Batch analysis
curl -X POST http://localhost:8000/api/v1/analyze-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello, how are you?", "Share your OTP now or account will be blocked"]}'

# Health check
curl http://localhost:8000/api/v1/health
```

---

## 📁 Project Structure

```
ScamGuard/
├── main.py                    # FastAPI backend (v5.0 — production-grade)
├── scamguard_enhanced.py      # Hybrid AI engine (v5.0 — optimized)
├── train_vishing_model.py     # Dataset builder + model trainer
├── vishing_data.csv           # Cleaned & balanced training dataset
├── logistic_vishing_model.pkl # Trained model (384-dim Multilingual-MiniLM)
├── index.html                 # Cinematic Neural Defense Grid UI (v5.0)
├── Dockerfile                 # Multi-stage Docker deployment
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable documentation
├── .gitignore                 # Git ignore rules
├── .dockerignore              # Docker build exclusions
├── LICENSE                    # MIT License
├── CHANGELOG.md               # Version history
└── README.md                  # You are here
```

---

## 🎮 Controls & Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Analyze current input |
| `Ctrl+K` | Open command palette |
| `Ctrl+M` | Toggle microphone |
| `Ctrl+V` | Jarvis voice mode |
| `Ctrl+R` | Reset session |
| `Ctrl+E` | Export report |
| `Ctrl+S` | Settings panel |
| `Ctrl+H` | Analysis history |
| `?` | Show all shortcuts |
| `1-4` | Load test scenarios |
| `↑↑↓↓←→←→BA` | Developer mode |

### Voice Commands (Jarvis Mode)
- **"Hey ScamGuard"** — wake word
- **"Analyze [text]"** — analyze a sentence
- **"Reset session"** — clear and restart
- **"Show report"** — export analysis report

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.13% |
| 5-Fold CV | 92.87% ±1.54% |
| Sanity Check | 15/16 (93.8%) |
| Safe Precision | 96% |
| Scam Recall | 97% |
| Embedding Model | Multilingual-MiniLM (384-dim) |
| Classifier | Logistic Regression |
| Avg Inference Latency | ~45ms |

---

## 👥 Team

- **Mohd Shaffan** — Lead Developer
- **Aditya Anurag Acharya** — AI/ML
- **Shaqueeb Jamil** — Research

**Manipal University Jaipur**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ and lots of ☕ | ScamGuard v5.0</sub>
</div>
