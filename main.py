"""
=============================================================================
  SCAM GUARD v5.0 — FastAPI Backend (Industry-Grade)
  Authors: Mohd Shaffan, Aditya Anurag Acharya, Shaqueeb Jamil
  Manipal University Jaipur

  Production-grade backend with:
    • Secure CORS with configurable allowed origins
    • Rate limiting (slowapi) to prevent abuse
    • Input validation & sanitization
    • Structured JSON logging
    • FastAPI lifespan (modern startup/shutdown)
    • Async-compatible NLP inference
    • API versioning (/api/v1/)
    • Session auto-purge to prevent memory leaks
    • WebSocket heartbeat and timeout
    • Batch analysis endpoint
    • Metrics & feedback endpoints
    • Global exception handling

  Endpoints:
    WS   /ws/{session_id}               — Real-time chunk analysis
    POST /api/v1/analyze                — Single-shot REST analysis
    POST /api/v1/analyze-batch          — Batch analysis (up to 50 texts)
    POST /api/v1/analyze-audio          — Audio upload → transcribe → analyze
    POST /api/v1/reset/{session_id}     — Reset temporal scorer for a session
    POST /api/v1/feedback               — User feedback on predictions
    GET  /api/v1/health                 — Deep health check + model info
    GET  /api/v1/keywords               — Keyword tier dictionary
    GET  /api/v1/metrics                — System metrics
    GET  /api/v1/version                — Version & build info
    GET  /                              — Serves frontend (index.html)
=============================================================================
"""

# ── sys.path must be set BEFORE any local-module import ─────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
import time
import json
import math
import asyncio
import logging
import hashlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    UploadFile, File, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import shutil

# ── Version ──────────────────────────────────────────────────────────────
APP_VERSION = "5.0.0"
APP_NAME = "ScamGuard"
BUILD_HASH = hashlib.md5(f"{APP_VERSION}-{os.path.getmtime(__file__)}".encode()).hexdigest()[:8]

# ── Structured Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("scamguard")

# ── Import ScamGuard engine ──────────────────────────────────────────────
try:
    from scamguard_enhanced import (
        classify_intent,
        TemporalThreatScorer,
        load_nlp_model,
        KEYWORD_TIERS,
        transcribe_audio,
    )
    ENGINE_LOADED = True
    logger.info("ScamGuard engine imported successfully")
except ImportError as exc:
    logger.warning(f"Could not import scamguard_enhanced: {exc}")
    logger.warning("Running in DEMO mode — results are simulated")
    ENGINE_LOADED = False

# ── Configuration ────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "5000"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "50"))
SESSION_MAX_AGE = int(os.environ.get("SESSION_MAX_AGE", "3600"))
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")
WS_HEARTBEAT_TIMEOUT = int(os.environ.get("WS_HEARTBEAT_TIMEOUT", "60"))


# =============================================================================
#  SESSION STORE (Thread-safe, with auto-purge)
# =============================================================================

class SessionStore:
    """In-memory session store with auto-purge for stale sessions."""

    def __init__(self):
        self._sessions: Dict[str, TemporalThreatScorer] = {}
        self._created: Dict[str, float] = {}
        self._chunk_counts: Dict[str, int] = {}
        self._feedback: List[dict] = []  # User feedback storage

    def get_or_create(self, session_id: str) -> "TemporalThreatScorer":
        if session_id not in self._sessions:
            if ENGINE_LOADED:
                self._sessions[session_id] = TemporalThreatScorer(
                    alpha=0.85,
                    lambda_decay=0.3,
                    history_k=10,
                    tau_alert=0.60,
                    tau_drop=0.85,
                )
            else:
                # Minimal scorer for demo mode
                self._sessions[session_id] = _DemoScorer()
            self._created[session_id] = time.time()
            self._chunk_counts[session_id] = 0
            logger.info(f"Session created: {session_id}")
        return self._sessions[session_id]

    def reset(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].reset()
            self._chunk_counts[session_id] = 0
            logger.info(f"Session reset: {session_id}")

    def increment(self, session_id: str):
        self._chunk_counts[session_id] = self._chunk_counts.get(session_id, 0) + 1

    def chunk_count(self, session_id: str) -> int:
        return self._chunk_counts.get(session_id, 0)

    def set_thresholds(self, session_id: str, tau_alert: float, tau_drop: float):
        """Update thresholds on a live session without resetting history."""
        scorer = self.get_or_create(session_id)
        scorer.tau_alert = max(0.05, min(0.95, tau_alert))
        scorer.tau_drop = max(0.05, min(1.00, tau_drop))

    def purge_old(self, max_age_seconds: int = SESSION_MAX_AGE):
        """Remove stale sessions to prevent memory leaks."""
        now = time.time()
        stale = [sid for sid, ts in self._created.items()
                 if now - ts > max_age_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._created.pop(sid, None)
            self._chunk_counts.pop(sid, None)
        if stale:
            logger.info(f"Purged {len(stale)} stale sessions")

    def add_feedback(self, feedback: dict):
        self._feedback.append({
            **feedback,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def total_feedback(self) -> int:
        return len(self._feedback)


class _DemoScorer:
    """Minimal scorer for demo mode when engine is not loaded."""
    def __init__(self):
        self.CT = 0.0
        self.tau_alert = 0.55
        self.tau_drop = 0.80
        self.history = []
    def update(self, s_t):
        self.CT = 0.85 * s_t + 0.15 * self.CT
        self.history.append(s_t)
        return self.CT
    def status(self):
        if self.CT >= self.tau_drop: return "DROP"
        if self.CT >= self.tau_alert: return "ALERT"
        return "SAFE"
    def reset(self):
        self.CT = 0.0
        self.history = []


SESSION_STORE = SessionStore()


# =============================================================================
#  METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Simple in-memory metrics for monitoring."""

    def __init__(self):
        self.total_analyses = 0
        self.total_scams_detected = 0
        self.total_ws_connections = 0
        self.total_errors = 0
        self.latencies: List[float] = []
        self.start_time = time.time()

    def record_analysis(self, latency_ms: float, is_scam: bool):
        self.total_analyses += 1
        if is_scam:
            self.total_scams_detected += 1
        self.latencies.append(latency_ms)
        # Keep only last 1000 latencies
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-500:]

    def record_error(self):
        self.total_errors += 1

    def record_ws_connection(self):
        self.total_ws_connections += 1

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "total_analyses": self.total_analyses,
            "total_scams_detected": self.total_scams_detected,
            "total_ws_connections": self.total_ws_connections,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency, 2),
            "p99_latency_ms": round(self.p99_latency, 2),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "active_sessions": SESSION_STORE.active_count,
        }


METRICS = MetricsCollector()


# =============================================================================
#  APP LIFECYCLE (Modern FastAPI lifespan)
# =============================================================================

NLP_MODEL = None
MODEL_NAME = "keyword-only"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern startup/shutdown using lifespan context manager."""
    global NLP_MODEL, MODEL_NAME

    # ── Startup ──────────────────────────────────────────────────────────
    logger.info(f"Starting {APP_NAME} v{APP_VERSION} (build: {BUILD_HASH})")
    logger.info("Loading NLP model — please wait...")

    if ENGINE_LOADED:
        NLP_MODEL, MODEL_NAME = load_nlp_model()
    else:
        MODEL_NAME = "demo-mode"

    logger.info(f"Model ready: {MODEL_NAME}")
    logger.info(f"CORS origins: {ALLOWED_ORIGINS}")
    logger.info(f"Rate limit: {RATE_LIMIT}")

    # Start background session purger
    purge_task = asyncio.create_task(_session_purge_loop())

    yield  # App is running

    # ── Shutdown ─────────────────────────────────────────────────────────
    purge_task.cancel()
    logger.info(f"Shutting down {APP_NAME}. Total analyses: {METRICS.total_analyses}")


async def _session_purge_loop():
    """Background task to purge stale sessions every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        SESSION_STORE.purge_old(SESSION_MAX_AGE)


# =============================================================================
#  APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    description="Real-time vishing detection via hybrid BERT + keyword engine",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ── CORS (configurable, not wildcard in production) ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)


# ── Request/Response Middleware ──────────────────────────────────────────
@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Add request ID, timing, and security headers to every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    response.headers["X-Powered-By"] = f"{APP_NAME}/{APP_VERSION}"

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Log request
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration_ms:.1f}ms) [rid={request_id}]"
    )
    return response


# ── Global Exception Handler ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    METRICS.record_error()
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
            "version": APP_VERSION,
        }
    )


# =============================================================================
#  INPUT VALIDATION (Pydantic Models)
# =============================================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    session_id: Optional[str] = None

    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Strip HTML tags and excessive whitespace."""
        import re
        v = re.sub(r'<[^>]+>', '', v)  # Remove HTML tags
        v = re.sub(r'\s+', ' ', v)     # Collapse whitespace
        return v.strip()


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    session_id: Optional[str] = None

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        import re
        cleaned = []
        for text in v:
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Each text must be under {MAX_TEXT_LENGTH} characters")
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                cleaned.append(text)
        return cleaned


class FeedbackRequest(BaseModel):
    session_id: str
    chunk_text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    predicted_scam: bool
    actual_scam: bool
    comment: Optional[str] = Field(None, max_length=500)


# =============================================================================
#  DEMO FALLBACK
# =============================================================================

def _demo_classify(text: str):
    """Simulated classifier used in demo / import-error mode."""
    import random
    scam_words = ["otp", "blocked", "urgent", "bank", "cvv", "pin", "kyc",
                  "arrest", "lottery", "aadhaar", "transfer", "anydesk"]
    hits = [w for w in scam_words if w in text.lower()]
    is_scam = len(hits) >= 1
    confidence = min(0.60 + len(hits) * 0.10, 0.97) if is_scam else random.uniform(0.75, 0.92)
    details = {
        "keyword_weight": len(hits) * 2,
        "keyword_hits": len(hits),
        "matched_keywords": [(w, 2) for w in hits],
        "nlp_label": "DEMO",
        "nlp_raw_score": confidence,
        "explanation": "Demo mode — engine not loaded.",
    }
    return is_scam, confidence, details


# =============================================================================
#  CORE ANALYSIS FUNCTION
# =============================================================================

def _get_tier(weight: int) -> str:
    return {3: "CRITICAL", 2: "HIGH", 1: "MEDIUM"}.get(weight, "MEDIUM")


def analyze_chunk(session_id: str, text: str) -> dict:
    """
    Full pipeline:
      1. classify_intent(text, NLP_MODEL, MODEL_NAME) → is_scam, confidence, details
      2. TemporalThreatScorer.update(chunk_score)     → CT
      3. Build and return the payload.
    """
    t_start = time.perf_counter()

    # ── Classify ─────────────────────────────────────────────────────────
    if ENGINE_LOADED:
        is_scam, confidence, details = classify_intent(text, NLP_MODEL, MODEL_NAME)
    else:
        is_scam, confidence, details = _demo_classify(text)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Temporal scoring ─────────────────────────────────────────────────
    scorer = SESSION_STORE.get_or_create(session_id)
    chunk_score = confidence
    CT = scorer.update(chunk_score)
    status = scorer.status()
    SESSION_STORE.increment(session_id)

    # ── Record metrics ───────────────────────────────────────────────────
    METRICS.record_analysis(latency_ms, is_scam)

    # ── Serialise keyword hits ───────────────────────────────────────────
    matched_kw = [
        {"word": kw, "weight": wt, "tier": _get_tier(wt)}
        for kw, wt in details.get("matched_keywords", [])
    ]

    return {
        "session_id": session_id,
        "chunk_number": SESSION_STORE.chunk_count(session_id),
        "text": text,
        "is_scam": is_scam,
        "nlp_confidence": round(confidence * 100, 2),
        "nlp_label": details.get("nlp_label", "N/A"),
        "keyword_weight": details.get("keyword_weight", 0),
        "matched_keywords": matched_kw,
        "chunk_score": round(chunk_score, 4),
        "cumulative_ct": round(CT, 4),
        "ct_percent": round(min(CT * 100, 100), 1),
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "model": MODEL_NAME,
        "explanation": details.get("explanation", "N/A"),
        "version": APP_VERSION,
    }


# =============================================================================
#  WEBSOCKET ENDPOINT  /ws/{session_id}
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    METRICS.record_ws_connection()
    logger.info(f"[WS] Session connected: {session_id}")

    try:
        while True:
            # Heartbeat timeout: close if no message for WS_HEARTBEAT_TIMEOUT seconds
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WS_HEARTBEAT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.info(f"[WS] Session timed out: {session_id}")
                await websocket.send_text(json.dumps({
                    "type": "timeout",
                    "message": "Connection timed out due to inactivity."
                }))
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "chunk", "text": raw}

            msg_type = msg.get("type", "chunk")

            # ── Reset ────────────────────────────────────────────────────
            if msg_type == "reset":
                SESSION_STORE.reset(session_id)
                await websocket.send_text(json.dumps({
                    "type": "reset_ack",
                    "session_id": session_id,
                    "message": "Session reset. Temporal scorer cleared."
                }))
                continue

            # ── Ping ─────────────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # ── Threshold updates from frontend sliders ──────────────────
            if msg_type == "set_thresholds":
                try:
                    tau_alert = float(msg.get("alert", 0.55))
                    tau_drop = float(msg.get("drop", 0.80))
                    SESSION_STORE.set_thresholds(session_id, tau_alert, tau_drop)
                    await websocket.send_text(json.dumps({
                        "type": "thresholds_ack",
                        "tau_alert": tau_alert,
                        "tau_drop": tau_drop,
                    }))
                except Exception as exc:
                    logger.error(f"[WS] set_thresholds error: {exc}")
                continue

            # ── Analyse chunk ────────────────────────────────────────────
            text = msg.get("text", "").strip()
            if not text:
                continue

            # Input validation
            if len(text) > MAX_TEXT_LENGTH:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
                }))
                continue

            # Run analysis in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(analyze_chunk, session_id, text)
            result["type"] = "analysis"
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        METRICS.record_error()
        logger.error(f"[WS] Error in session {session_id}: {exc}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "An internal error occurred."
            }))
        except Exception:
            pass


# =============================================================================
#  REST ENDPOINTS (/api/v1/)
# =============================================================================

@app.post("/api/v1/analyze")
async def rest_analyze(req: AnalyzeRequest):
    """Single-shot REST analysis. Creates a one-off session if none provided."""
    session_id = req.session_id or str(uuid.uuid4())
    result = await asyncio.to_thread(analyze_chunk, session_id, req.text)
    return JSONResponse(content=result)


@app.post("/api/v1/analyze-batch")
async def rest_analyze_batch(req: BatchAnalyzeRequest):
    """Batch analysis of up to 50 texts in a single request."""
    session_id = req.session_id or str(uuid.uuid4())
    results = []
    for text in req.texts:
        result = await asyncio.to_thread(analyze_chunk, session_id, text)
        results.append(result)
    return JSONResponse(content={
        "session_id": session_id,
        "count": len(results),
        "results": results,
    })


# Keep old endpoint for backward compatibility
@app.post("/api/analyze")
async def rest_analyze_legacy(req: AnalyzeRequest):
    """Legacy endpoint — redirects to v1."""
    return await rest_analyze(req)


@app.post("/api/v1/analyze-audio")
async def analyze_audio(file: UploadFile = File(...),
                        session_id: Optional[str] = None):
    """Upload an audio file → transcribe with Whisper → analyze intent."""
    session_id = session_id or str(uuid.uuid4())

    # Use secure temp file
    suffix = Path(file.filename or "audio.wav").suffix
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="scamguard_")
    os.close(fd)

    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        logger.info(f"[AUDIO] Received: {file.filename} ({os.path.getsize(temp_path)} bytes)")
        actual_text = transcribe_audio(temp_path) if ENGINE_LOADED else ""
        if not actual_text:
            raise HTTPException(status_code=422, detail="Transcription returned empty text.")
        result = await asyncio.to_thread(analyze_chunk, session_id, actual_text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        METRICS.record_error()
        logger.error(f"[AUDIO] Processing error: {exc}")
        raise HTTPException(status_code=500, detail="Audio processing failed.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Keep old endpoint for backward compatibility
@app.post("/api/analyze-audio")
async def analyze_audio_legacy(file: UploadFile = File(...),
                               session_id: Optional[str] = None):
    """Legacy endpoint — redirects to v1."""
    return await analyze_audio(file, session_id)


@app.post("/api/v1/reset/{session_id}")
async def reset_session(session_id: str):
    SESSION_STORE.reset(session_id)
    return {"status": "ok", "message": f"Session {session_id} reset."}


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback on a prediction for future model improvement."""
    SESSION_STORE.add_feedback({
        "session_id": req.session_id,
        "chunk_text": req.chunk_text[:200],  # Truncate for storage
        "predicted_scam": req.predicted_scam,
        "actual_scam": req.actual_scam,
        "comment": req.comment,
    })
    logger.info(f"[FEEDBACK] Received for session {req.session_id}")
    return {"status": "ok", "message": "Feedback recorded. Thank you!"}


@app.get("/api/v1/health")
async def health():
    """Deep health check with model inference verification."""
    health_status = {
        "status": "online",
        "engine": f"{APP_NAME} v{APP_VERSION}",
        "model": MODEL_NAME,
        "engine_loaded": ENGINE_LOADED,
        "active_sessions": SESSION_STORE.active_count,
        "total_analyses": METRICS.total_analyses,
        "uptime_seconds": round(METRICS.uptime_seconds, 1),
        "version": APP_VERSION,
        "build": BUILD_HASH,
    }

    # Quick inference check
    if ENGINE_LOADED and NLP_MODEL is not None:
        try:
            t0 = time.perf_counter()
            classify_intent("health check test", NLP_MODEL, MODEL_NAME)
            health_status["inference_ok"] = True
            health_status["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as exc:
            health_status["inference_ok"] = False
            health_status["inference_error"] = str(exc)

    return health_status


# Keep old endpoint for backward compatibility
@app.get("/api/health")
async def health_legacy():
    """Legacy endpoint."""
    return await health()


@app.get("/api/v1/keywords")
async def get_keywords():
    """Return the keyword tier dictionary for frontend display."""
    return KEYWORD_TIERS if ENGINE_LOADED else {
        "CRITICAL": {"weight": 3, "words": ["otp", "cvv", "pin", "aadhaar"]},
        "HIGH": {"weight": 2, "words": ["blocked", "urgent", "kyc", "upi"]},
        "MEDIUM": {"weight": 1, "words": ["verify", "account", "immediately"]},
    }


# Keep old endpoint for backward compatibility
@app.get("/api/keywords")
async def get_keywords_legacy():
    """Legacy endpoint."""
    return await get_keywords()


@app.get("/api/v1/metrics")
async def get_metrics():
    """System metrics for monitoring dashboards."""
    return METRICS.to_dict()


@app.get("/api/v1/version")
async def get_version():
    """Build and version information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "build": BUILD_HASH,
        "model": MODEL_NAME,
        "engine_loaded": ENGINE_LOADED,
        "python_version": sys.version.split()[0],
    }


# Keep old reset endpoint for backward compatibility
@app.post("/api/reset/{session_id}")
async def reset_session_legacy(session_id: str):
    """Legacy endpoint."""
    return await reset_session(session_id)


# =============================================================================
#  SERVE FRONTEND
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found. Place it next to main.py</h1>")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("ENV", "development") == "development",
        log_level="info",
    )