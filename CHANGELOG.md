# Changelog

All notable changes to ScamGuard will be documented in this file.

## [5.0.0] — 2026-04-16

### 🔒 Security
- Fixed CORS to use configurable allowed origins instead of wildcard `*`
- Added input validation & sanitization (max length, HTML stripping)
- Added security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy)
- Added secure temp file handling for audio uploads
- Added WebSocket heartbeat timeout (60s)
- Added CSP meta tag and SRI hashes for CDN resources

### 🏗️ Backend
- Migrated from deprecated `@app.on_event("startup")` to modern `lifespan` context manager
- Replaced all `print()` calls with structured `logging` module
- Added request/response middleware (request ID, timing headers)
- Made NLP inference async-compatible with `asyncio.to_thread()`
- Added API versioning: all endpoints now under `/api/v1/` (legacy endpoints preserved)
- Added batch analysis endpoint (`POST /api/v1/analyze-batch`)
- Added user feedback endpoint (`POST /api/v1/feedback`)
- Added metrics endpoint (`GET /api/v1/metrics`)
- Added version endpoint (`GET /api/v1/version`)
- Added deep health check with model inference verification
- Added background session purging (every 5 minutes, 1-hour TTL)
- Added global exception handler with JSON responses
- Added Pydantic input validation with `field_validator`

### 🧠 AI/ML Engine
- Added `EngineConfig` dataclass — all tunable parameters in one place
- Added text preprocessing pipeline (URL/phone/email masking, whitespace normalization)
- Added LRU embedding cache for repeated texts (512 entries)
- Pre-compiled all keyword regex patterns at module load time
- Cached Whisper model at module level (no longer reloads per call)
- Added model hash tracking for version reproducibility
- Added Hindi negation words ("mat batana", "nahi dena", "savdhan")
- Improved explainable AI to show which specific keywords were detected
- Added typed `ClassificationResult` dataclass

### 🎨 Frontend UI/UX
- Fixed fake accuracy display (removed `Math.random()` noise)
- Added real model accuracy from `/api/v1/health` endpoint
- Added analysis history with `localStorage` persistence
- Added keyword highlighting in analysis log entries
- Added loading skeleton while waiting for WebSocket response
- Added boot screen skip button and auto-skip after 5 seconds
- Added ARIA labels and roles for accessibility
- Added keyboard focus indicators (`:focus-visible`)
- Added Hindi/English UI language toggle
- Added favicon (shield emoji encoded as SVG)
- Added Open Graph meta tags for social sharing
- Added `document.hidden` check to pause Three.js when tab is hidden
- Reduced particles to 120 for better performance
- Added CSP meta tag
- Unified version strings to v5.0.0 across all UI elements

### 📦 DevOps
- Added multi-stage Docker build (smaller image)
- Added Docker HEALTHCHECK directive
- Created `.dockerignore` to reduce build context
- Created `.env.example` with all configuration options
- Created `LICENSE` (MIT)
- Updated `README.md` with accurate v5.0 information
- Removed stale `debug_pred.py` (used wrong model name)

### 📚 Documentation
- Updated README with v5.0 architecture, correct model info (384-dim multilingual)
- Added API documentation section
- Added deployment guide
- Created `CHANGELOG.md`

## [3.0.0] — 2026-04-15
- Cinematic Neural Defense Grid UI
- Three.js holographic environment
- Jarvis voice mode
- Command palette
- 5 color themes

## [2.0.0] — 2026-04-10
- Hybrid AI engine (DistilBERT + Logistic Regression)
- Temporal Threat Scoring algorithm
- WebSocket real-time streaming
- FastAPI backend

## [1.0.0] — 2026-04-01
- Initial prototype
- Basic keyword detection
- Simple web interface
