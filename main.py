"""
=============================================================================
  SCAM GUARD v2.0 — FastAPI Backend (LAZY LOADING FOR RENDER)
=============================================================================
"""

import uuid
import time
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from scamguard_enhanced import (
        classify_intent,
        TemporalThreatScorer,
        load_nlp_model,
        KEYWORD_TIERS,
    )
    ENGINE_LOADED = True
except ImportError as e:
    ENGINE_LOADED = False

app = FastAPI(title="ScamGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Lazy Loading
NLP_MODEL = None
MODEL_NAME = "Initializing... (Will load on first prompt)"

class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, TemporalThreatScorer] = {}
        self._chunk_counts: Dict[str, int] = {}

    def get_or_create(self, session_id: str) -> TemporalThreatScorer:
        if session_id not in self._sessions:
            self._sessions[session_id] = TemporalThreatScorer()
            self._chunk_counts[session_id] = 0
        return self._sessions[session_id]

    def reset(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].reset()
            self._chunk_counts[session_id] = 0

    def increment(self, session_id: str):
        self._chunk_counts[session_id] = self._chunk_counts.get(session_id, 0) + 1

    def chunk_count(self, session_id: str) -> int:
        return self._chunk_counts.get(session_id, 0)

SESSION_STORE = SessionStore()

def _demo_classify(text: str):
    scam_words = ["otp", "blocked", "urgent", "bank", "cvv"]
    hits = [w for w in scam_words if w in text.lower()]
    is_scam = len(hits) >= 1
    confidence = min(0.60 + len(hits) * 0.10, 0.97) if is_scam else random.uniform(0.75, 0.92)
    return is_scam, confidence, {"keyword_weight": len(hits) * 2, "matched_keywords": [(w, 2) for w in hits], "nlp_label": "DEMO"}

def analyze_chunk(session_id: str, text: str) -> dict:
    global NLP_MODEL, MODEL_NAME
    
    # LAZY LOADING LOGIC: Load model only when first message arrives!
    if NLP_MODEL is None and ENGINE_LOADED:
        print("[ScamGuard] First chunk received! Downloading & Loading AI Model now...")
        NLP_MODEL, MODEL_NAME = load_nlp_model()
    elif NLP_MODEL is None:
        MODEL_NAME = "demo-mode"

    t_start = time.perf_counter()

    if ENGINE_LOADED:
        is_scam, confidence, details = classify_intent(text, NLP_MODEL, MODEL_NAME)
    else:
        is_scam, confidence, details = _demo_classify(text)

    latency_ms = (time.perf_counter() - t_start) * 1000

    scorer = SESSION_STORE.get_or_create(session_id)
    chunk_score = confidence if is_scam else (1.0 - confidence)
    CT = scorer.update(chunk_score)
    status = scorer.status()
    SESSION_STORE.increment(session_id)

    matched_kw = [{"word": kw, "weight": wt, "tier": "HIGH"} for kw, wt in details.get("matched_keywords", [])]

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
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "chunk", "text": raw}

            if msg.get("type") == "reset":
                SESSION_STORE.reset(session_id)
                await websocket.send_text(json.dumps({"type": "reset_ack"}))
                continue
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            text = msg.get("text", "").strip()
            if text:
                result = analyze_chunk(session_id, text)
                result["type"] = "analysis"
                await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass

@app.get("/api/health")
async def health():
    return {"status": "online", "model": MODEL_NAME, "engine_loaded": ENGINE_LOADED}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8")) if html_path.exists() else HTMLResponse(content="index.html not found")
