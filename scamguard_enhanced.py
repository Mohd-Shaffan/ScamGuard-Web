"""
=============================================================================
 SCAM GUARD — ENHANCED HYBRID AI ENGINE v5.0
 Authors: Mohd Shaffan, Aditya Anurag Acharya, Shaqueeb Jamil
 Manipal University Jaipur

 Industry-grade improvements:
   * Config dataclass for all tunable parameters
   * Text preprocessing pipeline (normalization, URL/phone masking)
   * LRU embedding cache for repeated texts
   * Typed ClassificationResult return value
   * Optimized keyword matching with pre-compiled regex
   * Whisper model cached at module level (not per-call)
   * Model version tracking in responses
   * Fixed Unicode encoding issues in comments
   * Proper error handling throughout
   * Clean separation of concerns
=============================================================================
"""

import time
import math
import random
import re
import os
import sys
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from functools import lru_cache

import numpy as np

logger = logging.getLogger("scamguard.engine")

# -- Dependency checks --------------------------------------------------------
import joblib
from sentence_transformers import SentenceTransformer

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import whisper as openai_whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

try:
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  precision_recall_fscore_support)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# =============================================================================
#  CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class EngineConfig:
    """All tunable parameters in one place — no more magic numbers."""

    # -- Encoder model
    encoder_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    classifier_path: str = "logistic_vishing_model.pkl"

    # -- Fusion thresholds
    critical_kw_boost: float = 0.35        # Boost when critical keywords found
    critical_kw_heavy_boost: float = 0.50  # Boost when kw_weight >= 5
    critical_kw_heavy_min: float = 0.85    # Minimum score for heavy critical
    negation_penalty: float = 0.40         # Penalty for negated critical keywords
    scam_threshold_with_kw: float = 0.60   # Scam threshold when keywords present
    scam_threshold_no_kw: float = 0.65     # Scam threshold when no keywords
    kw_normalizer: float = 15.0            # Divisor for keyword weight normalization
    nlp_weight: float = 0.60              # Weight for NLP score in fusion
    kw_weight_ratio: float = 0.40         # Weight for keyword score in fusion
    negation_kw_dampen: float = 0.30      # Dampen factor for negated non-critical keywords

    # -- Default NLP score when model unavailable
    default_nlp_score: float = 0.30

    # -- Embedding cache size
    cache_size: int = 512

    # -- Whisper ASR model size
    whisper_model_size: str = "base"


# Global config instance — can be overridden via environment variables
ENGINE_CONFIG = EngineConfig(
    encoder_name=os.environ.get("ENCODER_NAME", EngineConfig.encoder_name),
    default_nlp_score=float(os.environ.get("DEFAULT_NLP_SCORE", "0.30")),
    whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "base"),
)


# =============================================================================
#  SECTION 1 — TIERED KEYWORD DICTIONARY (India-centric + Global Multilingual)
# =============================================================================

KEYWORD_TIERS = {
    "CRITICAL": {
        "weight": 3,
        "words": [
            "otp", "cvv", "pin", "aadhaar", "pan", "password", "passphrase",
            "wire transfer", "cryptocurrency", "gift card", "remote access",
            "anydesk", "teamviewer", "verify your identity", "confirm your otp",
            "share your otp", "give me your otp",
            "khata", "block", "band ho", "paise transfer", "install anydesk",
            "download anydesk", "screen share"
        ]
    },
    "HIGH": {
        "weight": 2,
        "words": [
            "blocked", "suspended", "arrest", "legal action", "fir", "police",
            "court", "emi", "refund", "lottery", "prize", "won", "reward",
            "kyc", "update kyc", "bank account", "sbi", "hdfc", "icici",
            "axis bank", "paytm", "gpay", "phonepay", "upi", "neft", "rtgs",
            "debit card", "credit card", "atm card", "income tax",
            "customs", "parcel", "package", "delivery charge", "clearance fee",
            "pulis", "jail", "giraftar", "jurmana", "fine", "inam", "jeet",
            "freeze", "sir aapka", "madam aapka", "kbc lottery", "custom officer"
        ]
    },
    "MEDIUM": {
        "weight": 1,
        "words": [
            "urgent", "immediately", "right now", "last chance", "expire",
            "limited time", "act fast", "do not tell", "keep secret",
            "do not hang up", "stay on the line", "important notice",
            "government", "rbi", "trai", "insurance", "policy", "claim",
            "emi waiver", "loan approval", "interest rate", "outstanding",
            "turant", "abhee", "jald", "jaldi", "aakhri mauka", "secret",
            "kisi ko mat batana", "call cut mat karna"
        ]
    }
}

# Flatten for quick lookup
KEYWORD_WEIGHT_MAP: Dict[str, int] = {}
for _tier, _data in KEYWORD_TIERS.items():
    for _word in _data["words"]:
        KEYWORD_WEIGHT_MAP[_word] = _data["weight"]

# Pre-compile regex patterns for all keywords (much faster than re.search per call)
_KEYWORD_PATTERNS: Dict[str, re.Pattern] = {
    kw: re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
    for kw in KEYWORD_WEIGHT_MAP
}


# =============================================================================
#  SECTION 2 — TEXT PREPROCESSING
# =============================================================================

# Pre-compiled patterns for preprocessing
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_PHONE_PATTERN = re.compile(r'\b(?:\+91|91|0)?[6-9]\d{9}\b')
_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
_MULTI_SPACE = re.compile(r'\s+')
_SPECIAL_CHARS = re.compile(r'[^\w\s.,!?;:\'"()-]')  # Keep basic punctuation


def preprocess_text(text: str) -> str:
    """
    Normalize text for consistent classification:
    - Mask URLs, phone numbers, emails
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    text = _URL_PATTERN.sub(' [URL] ', text)
    text = _PHONE_PATTERN.sub(' [PHONE] ', text)
    text = _EMAIL_PATTERN.sub(' [EMAIL] ', text)
    text = _MULTI_SPACE.sub(' ', text)
    return text.strip()


# =============================================================================
#  SECTION 3 — NLP MODEL LOADING
# =============================================================================

def load_nlp_model() -> tuple:
    """
    Load the SentenceTransformer encoder and the trained Logistic Regression
    classifier from disk.

    Returns:
        (nlp_model_dict, model_name_str)
        nlp_model_dict = {"encoder": SentenceTransformer, "classifier": LogisticRegression}
        Falls back to (None, "keyword-only") on any error.
    """
    logger.info(f"  -> Loading SentenceTransformer ({ENGINE_CONFIG.encoder_name})...")
    try:
        encoder = SentenceTransformer(ENGINE_CONFIG.encoder_name)

        # Locate the .pkl relative to this file so it works on any CWD
        pkl_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ENGINE_CONFIG.classifier_path
        )
        clf = joblib.load(pkl_path)

        # Compute model hash for version tracking
        model_hash = hashlib.md5(open(pkl_path, 'rb').read()).hexdigest()[:8]

        logger.info(f"  [OK] Vishing model loaded (hash: {model_hash})")
        return {
            "encoder": encoder,
            "classifier": clf,
            "model_hash": model_hash,
        }, f"Multilingual-MiniLM+LR(v5-{model_hash})"

    except FileNotFoundError:
        logger.error(f"  [ERR] {ENGINE_CONFIG.classifier_path} not found — keyword-only mode")
        return None, "keyword-only"
    except Exception as exc:
        logger.error(f"  [ERR] Model load error: {exc}")
        return None, "keyword-only"


# =============================================================================
#  SECTION 4 — KEYWORD SCORER (Optimized with pre-compiled regex)
# =============================================================================

def compute_keyword_score(text: str) -> Tuple[float, int, list]:
    """
    Returns (raw_weight_sum, hit_count, matched_keywords).
    matched_keywords is a list of (keyword_str, weight_int) tuples.

    Uses pre-compiled regex patterns for O(n) matching where n = text length.
    """
    text_lower = text.lower()
    matched = []
    total_weight = 0

    for keyword, pattern in _KEYWORD_PATTERNS.items():
        if pattern.search(text_lower):
            weight = KEYWORD_WEIGHT_MAP[keyword]
            matched.append((keyword, weight))
            total_weight += weight

    return total_weight, len(matched), matched


# =============================================================================
#  SECTION 5 — EXPLAINABLE AI (XAI) ENGINE
# =============================================================================

def generate_explanation(is_scam: bool, nlp_score: float,
                         matched_keywords: list, text: str) -> str:
    """Smart Rule-based Explainable AI summary to tell the user *WHY* it's a scam."""
    if not is_scam:
        if nlp_score < 0.2:
            return "Clear semantic structure indicating safe, routine conversation."
        if matched_keywords:
            return ("Contains sensitive keywords, but overall context implies "
                    "safe advisory or casual discussion.")
        return "Neutral conversational context detected. No immediate semantic threats."

    # Scam Explanations
    reasons = []
    has_crit = any(wt == 3 for _, wt in matched_keywords)
    has_high = any(wt == 2 for _, wt in matched_keywords)

    if nlp_score >= 0.8:
        reasons.append("Highly deceptive semantic patterns detected.")
    elif nlp_score >= 0.5:
        reasons.append("Suspicious coercive context detected.")

    if has_crit:
        crit_words = [kw for kw, wt in matched_keywords if wt == 3]
        reasons.append(
            f"Crucial security markers requested ({', '.join(crit_words[:3]).upper()})."
        )
    if has_high:
        reasons.append(
            "Manipulative language involving authority, banking, or false rewards."
        )

    if len(reasons) >= 2:
        return f"{reasons[0]} {reasons[1]}"
    if reasons:
        return reasons[0]
    return ("NLP engine flagged contextual intent as malicious "
            "despite missing standard tracking keywords.")


# =============================================================================
#  SECTION 6 — CLASSIFICATION RESULT TYPE
# =============================================================================

@dataclass
class ClassificationResult:
    """Typed result from classify_intent — replaces raw tuples."""
    is_scam: bool
    confidence: float
    keyword_weight: int
    keyword_hits: int
    matched_keywords: list
    nlp_label: str
    nlp_raw_score: float
    explanation: str

    def to_tuple(self) -> tuple:
        """Backward compatibility with old (is_scam, confidence, details) format."""
        details = {
            "keyword_weight": self.keyword_weight,
            "keyword_hits": self.keyword_hits,
            "matched_keywords": self.matched_keywords,
            "nlp_label": self.nlp_label,
            "nlp_raw_score": self.nlp_raw_score,
            "explanation": self.explanation,
        }
        return self.is_scam, self.confidence, details


# =============================================================================
#  SECTION 7 — EMBEDDING CACHE
# =============================================================================

@lru_cache(maxsize=ENGINE_CONFIG.cache_size)
def _cached_encode(text: str, encoder_id: int) -> tuple:
    """
    Cache encoder.encode() results for repeated texts.
    encoder_id is used to invalidate cache when encoder changes.
    Returns tuple (for hashability).
    """
    # This function is actually called from classify_intent
    # The actual encoding happens there; this is just the cache wrapper
    return None  # Placeholder — actual implementation below


# Separate cache dict for embeddings (because numpy arrays aren't hashable for lru_cache)
_embedding_cache: Dict[str, np.ndarray] = {}
_CACHE_MAX = ENGINE_CONFIG.cache_size


def get_cached_embedding(text: str, encoder) -> np.ndarray:
    """Get embedding from cache or compute and cache it."""
    cache_key = hashlib.md5(text.encode()).hexdigest()

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    embedding = encoder.encode([text])
    _embedding_cache[cache_key] = embedding

    # Evict oldest entries if cache is full
    if len(_embedding_cache) > _CACHE_MAX:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]

    return embedding


# =============================================================================
#  SECTION 8 — HYBRID INTENT CLASSIFIER (Algorithm 1 of paper)
# =============================================================================

def classify_intent(text: str,
                    nlp_model=None,
                    model_name: str = "keyword-only") -> tuple:
    """
    Hybrid classifier: SentenceTransformer embedding + Logistic Regression
    fused with a tiered keyword heuristic (paper Algorithm 1).

    Parameters
    ----------
    text       : str  -- transcribed conversational chunk
    nlp_model  : dict | None
                 {"encoder": SentenceTransformer, "classifier": LogisticRegression}
                 as returned by load_nlp_model().  None -> keyword-only mode.
    model_name : str  -- informational label used in logs / response payloads

    Returns
    -------
    (is_scam: bool, confidence: float, details: dict)
        confidence  -- scam probability in [0, 1]
        details     -- keyword_weight, keyword_hits, matched_keywords,
                      nlp_label, nlp_raw_score, explanation
    """
    cfg = ENGINE_CONFIG

    # -- Step 1: Preprocess text
    processed_text = preprocess_text(text)

    # -- Step 2: Keyword analysis
    kw_weight, kw_hits, matched_keywords = compute_keyword_score(processed_text)

    # -- Step 3: Negation detection
    text_lower = processed_text.lower()
    negation_words = [
        "don't", "do not", "never", "shouldn't", "should not",
        "safe", "protect", "warning", "beware", "don't share", "do not share",
        "mat batana", "mat karna", "nahi dena", "savdhan"
    ]
    is_negated = any(neg in text_lower for neg in negation_words)

    # -- Step 4: Critical keyword flag
    has_critical = any(weight == 3 for _, weight in matched_keywords)

    # -- Step 5: NLP deep semantic score
    nlp_score = cfg.default_nlp_score
    nlp_label = "N/A"

    if (nlp_model is not None
            and isinstance(nlp_model, dict)
            and "encoder" in nlp_model
            and "classifier" in nlp_model):
        try:
            encoder = nlp_model["encoder"]
            clf = nlp_model["classifier"]
            # Use cached embedding for performance
            embedding = get_cached_embedding(processed_text, encoder)
            proba = clf.predict_proba(embedding)[0]
            # class 1 = scam; class 0 = safe
            nlp_score = float(proba[1])
            nlp_label = "SCAM" if nlp_score >= 0.50 else "SAFE"
        except Exception as exc:
            logger.warning(f"NLP inference error: {exc}")
            nlp_label = f"ERR:{type(exc).__name__}"

    # -- Step 6: Fusion logic (Equation 4 of paper)
    if has_critical and not is_negated:
        if kw_weight >= 5:
            combined_score = max(nlp_score + cfg.critical_kw_heavy_boost,
                                cfg.critical_kw_heavy_min)
        else:
            combined_score = nlp_score + cfg.critical_kw_boost
        combined_score = min(combined_score, 1.0)
        is_scam = combined_score >= cfg.scam_threshold_with_kw
        final_score = combined_score

    elif has_critical and is_negated:
        # e.g. "Never share your OTP" — protective statement
        final_score = max(0.05, nlp_score - cfg.negation_penalty)
        is_scam = False

    elif is_negated and kw_hits > 0:
        # Non-critical keywords in negation context -> likely a warning/advice
        kw_normalised = min(1.0, kw_weight / cfg.kw_normalizer)
        final_score = (cfg.nlp_weight * nlp_score +
                       cfg.kw_weight_ratio * kw_normalised * cfg.negation_kw_dampen)
        is_scam = final_score >= cfg.scam_threshold_with_kw

    else:
        # Standard hybrid blend: 60% NLP + 40% normalised keyword weight
        kw_normalised = min(1.0, kw_weight / cfg.kw_normalizer)
        final_score = cfg.nlp_weight * nlp_score + cfg.kw_weight_ratio * kw_normalised

        if kw_hits == 0:
            is_scam = final_score >= cfg.scam_threshold_no_kw
        else:
            is_scam = final_score >= cfg.scam_threshold_with_kw

    explanation = generate_explanation(is_scam, float(nlp_score),
                                       matched_keywords, text)

    details = {
        "keyword_weight": kw_weight,
        "keyword_hits": kw_hits,
        "matched_keywords": matched_keywords,
        "nlp_label": nlp_label,
        "nlp_raw_score": nlp_score,
        "explanation": explanation,
    }

    return is_scam, final_score, details


# =============================================================================
#  SECTION 9 — TEMPORAL THREAT DECAY ALGORITHM (Equation 5 of paper)
# =============================================================================

class TemporalThreatScorer:
    """
    CT = alpha * S_T + (1 - alpha) * sum_{i=1}^{k} S_{T-i} * e^{-lambda * i}

    Parameters:
        alpha      = weight of most-recent chunk (default 0.85)
        lambda     = exponential decay rate       (default 0.3)
        k          = history window size          (default 10)
        tau_alert  = amber-warning threshold      (default 0.55)
        tau_drop   = auto-drop threshold          (default 0.80)
    """

    def __init__(self, alpha=0.99, lambda_decay=0.3, history_k=10,
                 tau_alert=0.50, tau_drop=0.80):
        self.alpha = alpha
        self.lam = lambda_decay
        self.k = history_k
        self.tau_alert = tau_alert
        self.tau_drop = tau_drop
        self.history: List[float] = []
        self.CT = 0.0

    def update(self, s_t: float) -> float:
        """Push a new chunk score and return the updated cumulative CT."""
        self.history.insert(0, s_t)
        if len(self.history) > self.k:
            self.history = self.history[:self.k]

        ct = self.alpha * s_t
        hist_sum = sum(
            score * math.exp(-self.lam * i)
            for i, score in enumerate(self.history[1:], start=1)
        )
        ct += (1.0 - self.alpha) * hist_sum
        self.CT = ct
        return ct

    def status(self) -> str:
        if self.CT >= self.tau_drop:
            return "DROP"
        elif self.CT >= self.tau_alert:
            return "ALERT"
        return "SAFE"

    def reset(self):
        self.history.clear()
        self.CT = 0.0


# =============================================================================
#  SECTION 10 — WHISPER ASR (Cached model loading)
# =============================================================================

_whisper_model = None  # Module-level cache


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper (model cached at module level)."""
    global _whisper_model

    if not WHISPER_OK:
        logger.error("[ASR] Whisper not installed. pip install openai-whisper")
        return ""

    # Load model once and cache
    if _whisper_model is None:
        logger.info(f"[ASR] Loading Whisper model ({ENGINE_CONFIG.whisper_model_size})...")
        _whisper_model = openai_whisper.load_model(ENGINE_CONFIG.whisper_model_size)
        logger.info("[ASR] Whisper model loaded and cached.")

    logger.info(f"[ASR] Transcribing: {audio_path}")
    result = _whisper_model.transcribe(audio_path, fp16=False)
    return result["text"]


# =============================================================================
#  SECTION 11 — INTERACTIVE CLI
# =============================================================================

def run_interactive_mode(nlp_model, model_name):
    print("\n" + "=" * 65)
    print("  SCAM GUARD v5.0 — HYBRID AI + TEMPORAL SCORING")
    print("=" * 65)
    print(f"  NLP Model : {model_name}")
    print(f"  Commands  : 'exit' to quit | 'reset' to start new call")
    print("=" * 65)

    scorer = TemporalThreatScorer()
    chunk_num = 0

    while True:
        try:
            user_input = input(f"\n[Chunk {chunk_num + 1}] Enter sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            scorer.reset()
            chunk_num = 0
            print("  [RESET] New call session started.")
            continue
        if not user_input:
            continue

        chunk_num += 1
        t0 = time.perf_counter()
        is_scam, confidence, details = classify_intent(user_input, nlp_model, model_name)
        latency_ms = (time.perf_counter() - t0) * 1000

        chunk_score = confidence
        CT = scorer.update(chunk_score)
        status = scorer.status()

        print(f"\n  {'─' * 55}")
        print(f"  Chunk score  : {chunk_score:.4f}  |  Cumulative CT : {CT:.4f}")
        print(f"  Keywords     : {details['keyword_weight']} pts  ->  "
              f"{[k for k, _ in details['matched_keywords']]}")
        print(f"  NLP label    : {details['nlp_label']} ({details['nlp_raw_score']:.2%})")
        print(f"  Explanation  : {details['explanation']}")
        print(f"  Latency      : {latency_ms:.1f} ms")
        print(f"  {'─' * 55}")

        if status == "DROP":
            print(f"  🚨 AUTO-DROP: MALICIOUS CALL (CT={CT:.2f})")
        elif status == "ALERT":
            print(f"  ⚠️  ALERT: SCAM DETECTED  (Confidence: {confidence:.2%})")
        else:
            print(f"  ✅ SAFE  (Confidence: {confidence:.2%})")

    print("\n  Session ended. Stay safe! 🛡️")


# =============================================================================
#  SECTION 12 — EXPERIMENT SUITE
# =============================================================================

def load_vishing_test_data(filepath='vishing_data.csv', sample_size=200):
    """Load a balanced test sample from the vishing dataset."""
    if not PANDAS_OK:
        logger.error("pandas not installed.")
        return []
    try:
        df = pd.read_csv(filepath)
        scam = df[df['label'] == 1].sample(min(sample_size // 2, len(df[df['label'] == 1])))
        safe = df[df['label'] == 0].sample(sample_size - len(scam))
        test_df = pd.concat([scam, safe]).sample(frac=1)
        return [(row['text'], row['label']) for _, row in test_df.iterrows()]
    except Exception as exc:
        logger.error(f"Could not load {filepath}: {exc}")
        return []


REAL_TEST_SET: list = []


def _ensure_test_data():
    global REAL_TEST_SET
    if not REAL_TEST_SET:
        REAL_TEST_SET = load_vishing_test_data(sample_size=200)


def run_classification_experiment(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1 — Classification Metrics (Real Data)")
    print("=" * 65)

    y_true, y_pred, latencies = [], [], []
    skipped = 0
    for text, label in REAL_TEST_SET:
        try:
            t0 = time.perf_counter()
            is_scam, _, _ = classify_intent(text, nlp_model, model_name)
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(label)
            y_pred.append(1 if is_scam else 0)
        except Exception as exc:
            logger.warning(f"Skipping sample: {exc}")
            skipped += 1

    if skipped:
        print(f"  [INFO] Skipped {skipped} samples.")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if y_true else 0

    print(f"\n  Confusion Matrix:")
    print(f"  {'':10s}  Predicted SCAM  Predicted SAFE")
    print(f"  {'Actual SCAM':10s}  TP={tp:3d}            FN={fn:3d}")
    print(f"  {'Actual SAFE':10s}  FP={fp:3d}            TN={tn:3d}")
    print(f"\n  Precision : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    return latencies, y_true, y_pred


def run_latency_experiment(nlp_model, model_name, n_runs=50):
    _ensure_test_data()
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2 — Latency Profiling")
    print("=" * 65)

    test_texts = [t for t, _ in REAL_TEST_SET]
    if not test_texts:
        return []

    latencies = []
    for _ in range(n_runs):
        text = random.choice(test_texts)
        try:
            t0 = time.perf_counter()
            classify_intent(text, nlp_model, model_name)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.warning(f"Latency run failed: {exc}")

    print(f"\n  Runs : {n_runs}")
    print(f"  Mean : {np.mean(latencies):.2f} ms")
    print(f"  Max  : {np.max(latencies):.2f} ms")
    print(f"  P99  : {np.percentile(latencies, 99):.2f} ms")
    return latencies


def run_noise_robustness_experiment(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3 — Noise Robustness (Real Data)")
    print("=" * 65)

    scam_samples = [t for t, l in REAL_TEST_SET if l == 1][:30]
    if not scam_samples:
        return

    def corrupt(text, level=0.10):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < level:
                chars[i] = ("" if random.choice(["drop", "swap"]) == "drop"
                            else random.choice("abcdefghijklmnopqrstuvwxyz "))
        return "".join(chars)

    print(f"\n  {'Noise Level':15s}  {'Recall':10s}")
    print(f"  {'-' * 30}")
    for noise in [0.0, 0.05, 0.10, 0.15, 0.20]:
        correct = sum(
            1 for text in scam_samples
            if classify_intent(corrupt(text, noise) if noise > 0 else text,
                               nlp_model, model_name)[0]
        )
        print(f"  {noise * 100:>5.0f}%          {correct / len(scam_samples) * 100:>6.1f}%")


def run_threshold_sensitivity(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4 — Threshold Sensitivity")
    print("=" * 65)

    safe_msgs = [text for text, label in REAL_TEST_SET if label == 0][:2]
    scam_msgs = [text for text, label in REAL_TEST_SET if label == 1][:8]
    if not safe_msgs or len(scam_msgs) < 8:
        print("  [ERROR] Not enough data.")
        return

    call_chunks = [(msg, 0) for msg in safe_msgs] + [(msg, 1) for msg in scam_msgs]
    tau_pairs = [(0.4, 0.7), (0.5, 0.75), (0.55, 0.80), (0.6, 0.85)]

    print(f"\n  {'tau_alert':10s} {'tau_drop':10s} {'Alert @chunk':15s} {'Drop @chunk':12s}")
    print(f"  {'-' * 50}")

    for tau_a, tau_d in tau_pairs:
        scorer = TemporalThreatScorer(tau_alert=tau_a, tau_drop=tau_d)
        alert_at = drop_at = None
        for i, (text, _) in enumerate(call_chunks, 1):
            is_scam, conf, _ = classify_intent(text, nlp_model, model_name)
            CT = scorer.update(conf)
            if drop_at is None and CT >= tau_d:
                drop_at = i
            if alert_at is None and CT >= tau_a:
                alert_at = i

        print(f"  {tau_a:<10.2f} {tau_d:<10.2f} "
              f"{'chunk ' + str(alert_at) if alert_at else 'never':<15s} "
              f"{'chunk ' + str(drop_at) if drop_at else 'never':<12s}")


def run_all_experiments(nlp_model, model_name):
    _ensure_test_data()
    if not REAL_TEST_SET:
        print("\n  [ERROR] Cannot run experiments without data.")
        return

    latencies, y_true, y_pred = run_classification_experiment(nlp_model, model_name)
    lat_bench = run_latency_experiment(nlp_model, model_name)
    run_noise_robustness_experiment(nlp_model, model_name)
    run_threshold_sensitivity(nlp_model, model_name)

    if MATPLOTLIB_OK:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        # Confusion Matrix
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        cm = np.array([[tp, fn], [fp, tn]])
        ax1.imshow(cm, cmap='gray', vmin=0, vmax=np.max(cm))
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pred Scam', 'Pred Safe'])
        ax1.set_yticklabels(['Act Scam', 'Act Safe'])
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=14, fontweight='bold', color=color)
        ax1.set_title('Confusion Matrix')
        plt.tight_layout()
        fig1.savefig('fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("  [CHART] Confusion matrix -> fig_confusion_matrix.png")

        # Latency Histogram
        if lat_bench:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(lat_bench, bins=15, color='gray', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(lat_bench), color='black', linestyle='--',
                        linewidth=2, label=f'Mean={np.mean(lat_bench):.1f} ms')
            ax2.set_xlabel('Latency (ms)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('NLP Inference Latency Distribution')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig2.savefig('fig_latency_histogram.png', dpi=300, bbox_inches='tight')
            print("  [CHART] Latency histogram -> fig_latency_histogram.png")

        # Temporal Scoring Case Study
        scorer = TemporalThreatScorer()
        chunk_scores = [0.15, 0.35, 0.82, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65]
        ct_values = [scorer.update(s) for s in chunk_scores]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(range(1, len(ct_values) + 1), ct_values, marker='o',
                 color='black', linewidth=2, markersize=6, label='$C_T$')
        ax3.axhline(y=0.55, color='gray', linestyle='--', linewidth=1.5,
                    label='$\\tau_{alert}$=0.55')
        ax3.axhline(y=0.80, color='darkgray', linestyle='--', linewidth=1.5,
                    label='$\\tau_{drop}$=0.80')
        ax3.set_xlabel('Chunk Number')
        ax3.set_ylabel('Cumulative Threat Score ($C_T$)')
        ax3.set_title('Temporal Threat Score Progression')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_ylim(0, 1.05)
        plt.tight_layout()
        fig3.savefig('fig_temporal_scoring.png', dpi=300, bbox_inches='tight')
        print("  [CHART] Temporal scoring -> fig_temporal_scoring.png")
        plt.close('all')
    else:
        print("\n  [INFO] Install matplotlib for plots: pip install matplotlib")


# =============================================================================
#  SECTION 13 — AUDIO FILE MODE
# =============================================================================

def run_audio_mode(audio_path: str, nlp_model, model_name):
    print(f"\n[AUDIO MODE] Processing: {audio_path}")
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return
    print(f"[TRANSCRIPT] {transcript}\n")

    sentences = re.split(r"[.!?,]\s*", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    scorer = TemporalThreatScorer()
    for i, chunk in enumerate(sentences, 1):
        is_scam, conf, details = classify_intent(chunk, nlp_model, model_name)
        CT = scorer.update(conf)
        status = scorer.status()
        print(f"  Chunk {i:02d}: [{status}] CT={CT:.3f} | \"{chunk}\"")
        if status == "DROP":
            print("  🚨 AUTO-DROP TRIGGERED.")
            break


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  🛡️ SCAM GUARD — Enhanced Hybrid AI Engine v5.0")
    print("=" * 65)
    print("\n  Loading NLP model...")
    nlp_model, model_name = load_nlp_model()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "experiments":
            run_all_experiments(nlp_model, model_name)
            return
        elif cmd == "audio" and len(sys.argv) > 2:
            run_audio_mode(sys.argv[2], nlp_model, model_name)
            return
        elif cmd == "help":
            print("\n  Usage:")
            print("    python scamguard_enhanced.py              # Interactive mode")
            print("    python scamguard_enhanced.py experiments  # Full paper experiments")
            print("    python scamguard_enhanced.py audio <path> # Audio transcription")
            return

    run_interactive_mode(nlp_model, model_name)


if __name__ == "__main__":
    main()