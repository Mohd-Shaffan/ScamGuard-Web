"""
=============================================================================
 SCAM GUARD - ENHANCED HYBRID AI ENGINE v2.0
 Authors: Mohd Shaffan, Aditya Anurag Acharya, Shaqueeb Jamil
 Manipal University Jaipur
 
 UPGRADES OVER DEMO:
  1. Weighted, tiered keyword dictionary (critical / high / medium risk)
  2. Temporal Threat Decay Algorithm (per Section IV-D of paper)
  3. Better base model: distilbert-base-uncased-finetuned-sst-2-english
     (swap for a fine-tuned spam model when available)
  4. Optional Whisper ASR for live audio transcription (Module 2)
  5. Full experiment suite from Section V & VI of the paper:
     - Confusion matrix, precision, recall, F1
     - Latency profiling
     - Noise injection robustness test
     - Threshold sensitivity analysis
=============================================================================
"""

import time
import math
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─── DEPENDENCY CHECK ────────────────────────────────────────────────────────
try:
    from transformers import pipeline
    TRANSFORMERS_OK = True
except ImportError:
    print("[WARN] transformers not installed. Run: pip install transformers torch")
    TRANSFORMERS_OK = False

try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False  # ASR disabled — text-only mode

try:
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  precision_recall_fscore_support)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")           # non-GUI backend for headless runs
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# =============================================================================
#  SECTION 1 — TIERED KEYWORD DICTIONARY  (India-centric + Global)
# =============================================================================
# Each tier has a weight used in the hybrid score.
# CRITICAL (3 pts) — these alone should trigger suspicion
# HIGH     (2 pts)
# MEDIUM   (1 pt)

KEYWORD_TIERS = {
    "CRITICAL": {
        "weight": 3,
        "words": [
            "otp", "cvv", "pin", "aadhaar", "pan", "password", "passphrase",
            "wire transfer", "cryptocurrency", "gift card", "remote access",
            "anydesk", "teamviewer", "verify your identity", "confirm your otp",
            "share your otp", "give me your otp"
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
            "customs", "parcel", "package", "delivery charge", "clearance fee"
        ]
    },
    "MEDIUM": {
        "weight": 1,
        "words": [
            "urgent", "immediately", "right now", "last chance", "expire",
            "limited time", "act fast", "do not tell", "keep secret",
            "do not hang up", "stay on the line", "important notice",
            "government", "rbi", "trai", "insurance", "policy", "claim",
            "emi waiver", "loan approval", "interest rate", "outstanding"
        ]
    }
}

# Flatten for quick lookup with weights
KEYWORD_WEIGHT_MAP = {}
for tier, data in KEYWORD_TIERS.items():
    for word in data["words"]:
        KEYWORD_WEIGHT_MAP[word] = data["weight"]


# =============================================================================
#  SECTION 2 — NLP MODEL LOADING
# =============================================================================

def load_nlp_model():
    """
    Load the best available spam/intent classifier.
    Priority: fine-tuned SMS spam → sentiment proxy → fallback heuristic-only.
    
    For production, replace model_name with a fine-tuned model on Indian scam
    transcripts (e.g., your own fine-tuned BERT from Section V of the paper).
    """
    models_to_try = [
        # Best option — fine-tuned SMS spam BERT (small but decent)
        ("mrm8488/bert-tiny-finetuned-sms-spam-detection", "text-classification"),
        # Fallback — DistilBERT sentiment (negative ≈ threatening tone)
        ("distilbert-base-uncased-finetuned-sst-2-english", "text-classification"),
    ]
    for model_name, task in models_to_try:
        try:
            print(f"  → Trying model: {model_name}")
            clf = pipeline(task, model=model_name)
            print(f"  ✓ Loaded: {model_name}")
            return clf, model_name
        except Exception as e:
            print(f"  ✗ Failed ({e})")
    print("  [FALLBACK] No NLP model loaded — running keyword-only mode.")
    return None, "keyword-only"


# =============================================================================
#  SECTION 3 — HYBRID KEYWORD + NLP SCORER
# =============================================================================

def compute_keyword_score(text: str) -> tuple[float, int, list]:
    """
    Returns (raw_weight_sum, hit_count, matched_keywords).
    Single CRITICAL word → weight 3, which already signals high risk.
    """
    text_lower = text.lower()
    matched = []
    total_weight = 0
    for keyword, weight in KEYWORD_WEIGHT_MAP.items():
        if keyword in text_lower:
            matched.append((keyword, weight))
            total_weight += weight
    hit_count = len(matched)
    return total_weight, hit_count, matched


def classify_intent(text: str, nlp_model, model_name: str) -> tuple[bool, float, dict]:
    """
    Two-pass hybrid classification (Section IV-C of paper).

    Pass 1 — NLP model prediction (P_NLP).
    Pass 2 — Keyword heuristic score (P_KW).
    
    Final logic:
      - CRITICAL keyword alone → scam (score boosted to ≥ 0.95)
      - NLP flags scam OR keyword weight ≥ 4 → scam
      - Keyword weight ≥ 6 → scam even if NLP says safe
    
    Returns (is_scam, confidence, details_dict)
    """
    kw_weight, kw_hits, kw_matches = compute_keyword_score(text)
    details = {
        "keyword_weight": kw_weight,
        "keyword_hits": kw_hits,
        "matched_keywords": kw_matches,
        "nlp_label": "N/A",
        "nlp_raw_score": 0.0,
    }

    # ── NLP Pass ──────────────────────────────────────────────────────────
    nlp_is_scam = False
    nlp_conf = 0.5

    if nlp_model is not None:
        try:
            result = nlp_model(text[:512])[0]   # BERT max 512 tokens
            label = result["label"].upper()
            score = result["score"]
            details["nlp_label"] = label
            details["nlp_raw_score"] = score

            # bert-tiny spam model uses LABEL_1 for spam
            # distilbert SST uses NEGATIVE for threatening tone
            if "LABEL_1" in label or "SPAM" in label or "NEGATIVE" in label:
                nlp_is_scam = True
                nlp_conf = score
            else:
                nlp_conf = 1.0 - score      # invert confidence for safe label
        except Exception:
            pass

    # ── Fusion Logic ──────────────────────────────────────────────────────
    # Check for any CRITICAL keyword (weight=3 per word)
    critical_hit = any(w == 3 for _, w in kw_matches)

    if critical_hit:
        # Single critical keyword alone is enough
        final_score = max(nlp_conf if nlp_is_scam else 0.85, 0.95)
        return True, final_score, details

    if nlp_is_scam and kw_weight >= 2:
        final_score = max(nlp_conf, 0.90)
        return True, final_score, details

    if nlp_is_scam:
        final_score = nlp_conf
        return True, final_score, details

    if kw_weight >= 6:
        # Very high keyword density — scam even if NLP missed
        final_score = min(0.70 + (kw_weight * 0.02), 0.95)
        return True, final_score, details

    if kw_weight >= 4:
        final_score = min(0.60 + (kw_weight * 0.02), 0.85)
        return True, final_score, details

    # Safe
    final_score = max(1.0 - nlp_conf - (kw_weight * 0.05), 0.50)
    return False, final_score, details


# =============================================================================
#  SECTION 4 — TEMPORAL THREAT DECAY ALGORITHM  (Equation 9 of paper)
# =============================================================================

class TemporalThreatScorer:
    """
    Implements the cumulative threat score:

        CT = α·S_T + (1-α) · Σ S_{T-i} · e^{-λi}

    α    = weight of most recent chunk (default 0.4)
    λ    = decay rate (default 0.3)
    k    = history window size (default 10 chunks)
    τ_alert = alert threshold (default 0.55)
    τ_drop  = auto-drop threshold (default 0.80)
    """

    def __init__(self, alpha=0.4, lambda_decay=0.3, history_k=10,
                 tau_alert=0.55, tau_drop=0.80):
        self.alpha = alpha
        self.lam = lambda_decay
        self.k = history_k
        self.tau_alert = tau_alert
        self.tau_drop = tau_drop
        self.history: list[float] = []     # scores, newest first
        self.CT = 0.0

    def update(self, s_t: float) -> float:
        """Push a new chunk score and return updated CT."""
        self.history.insert(0, s_t)
        if len(self.history) > self.k:
            self.history = self.history[:self.k]

        # Current chunk
        ct = self.alpha * s_t

        # Historical weighted sum
        hist_sum = 0.0
        for i, score in enumerate(self.history[1:], start=1):
            hist_sum += score * math.exp(-self.lam * i)

        ct += (1 - self.alpha) * hist_sum
        self.CT = ct
        return ct

    def status(self) -> str:
        if self.CT >= self.tau_drop:
            return "DROP"
        elif self.CT >= self.tau_alert:
            return "ALERT"
        else:
            return "SAFE"

    def reset(self):
        self.history.clear()
        self.CT = 0.0


# =============================================================================
#  SECTION 5 — OPTIONAL WHISPER ASR MODULE  (Module 2 of paper)
# =============================================================================

def transcribe_audio(audio_path: str) -> str:
    """
    Use Whisper to transcribe an audio file.
    Install: pip install openai-whisper
    Model sizes: tiny, base, small, medium, large
    """
    if not WHISPER_OK:
        print("[ASR] Whisper not installed. pip install openai-whisper")
        return ""
    print(f"[ASR] Transcribing: {audio_path}")
    model = whisper.load_model("base")   # "small" for better accuracy
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


# =============================================================================
#  SECTION 6 — INTERACTIVE CLI (Text Mode)
# =============================================================================

def run_interactive_mode(nlp_model, model_name):
    """
    Interactive session with full temporal scoring.
    Simulates a live call — each input is one 'chunk' of the conversation.
    """
    print("\n" + "="*65)
    print("  🛡  SCAM GUARD v2.0  —  HYBRID AI + TEMPORAL SCORING  🛡")
    print("="*65)
    print(f"  NLP Model : {model_name}")
    print(f"  Mode      : Text chunks (simulate live call sentences)")
    print(f"  Commands  : 'exit' to quit | 'reset' to start new call")
    print("="*65)

    scorer = TemporalThreatScorer()
    chunk_num = 0

    while True:
        try:
            user_input = input(f"\n[Chunk {chunk_num+1}] Enter sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            scorer.reset()
            chunk_num = 0
            print("  🔄  New call session started.")
            continue
        if not user_input:
            continue

        chunk_num += 1
        t0 = time.perf_counter()
        is_scam, confidence, details = classify_intent(user_input, nlp_model, model_name)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Convert binary scam to a 0–1 score for temporal scorer
        chunk_score = confidence if is_scam else (1.0 - confidence)
        CT = scorer.update(chunk_score)
        status = scorer.status()

        # ── Display ──────────────────────────────────────────────
        print(f"\n  {'─'*55}")
        print(f"  Chunk score  : {chunk_score:.4f}  |  Cumulative CT : {CT:.4f}")
        print(f"  Keywords     : {details['keyword_weight']} pts  →  "
              f"{[k for k,_ in details['matched_keywords']]}")
        print(f"  NLP label    : {details['nlp_label']} ({details['nlp_raw_score']:.2%})")
        print(f"  Latency      : {latency_ms:.1f} ms")
        print(f"  {'─'*55}")

        if status == "DROP":
            print(f"  🚨  AUTO-DROP: MALICIOUS CALL (CT={CT:.2f})")
        elif status == "ALERT":
            if is_scam:
                print(f"  🔴  ALERT: SCAM DETECTED  (Confidence: {confidence:.2%})")
            else:
                print(f"  🟡  CAUTION: Suspicious pattern building (CT={CT:.2f})")
        else:
            print(f"  ✅  SAFE  (Confidence: {confidence:.2%})")

    print("\n  Session ended. Stay safe! 🛡")


# =============================================================================
#  SECTION 7 — EXPERIMENT SUITE  (Sections V & VI of paper)
# =============================================================================

# ── 7A. Synthetic test dataset ────────────────────────────────────────────────
SYNTHETIC_TEST_SET = [
    # (text, true_label)  1=scam, 0=safe
    # --- Scam (should be 1) ---
    ("Sir your HDFC account is blocked. Share your OTP immediately.", 1),
    ("Your SBI debit card is suspended. Call us to update KYC.", 1),
    ("Congratulations! You have won a lottery of 50 lakhs. Send your Aadhaar and PAN.", 1),
    ("This is RBI. Your account shows suspicious activity. Give your CVV to verify.", 1),
    ("Your parcel is held at customs. Pay Rs 2000 clearance fee via UPI.", 1),
    ("Income tax department. A case is filed against you. Share your PIN to avoid arrest.", 1),
    ("plz share your otp", 1),                          # THIS WAS MISSED BEFORE
    ("Do not tell anyone. Wire transfer Rs 50000 to this account urgently.", 1),
    ("Your KYC is expired. Update now or your account will be permanently blocked.", 1),
    ("Microsoft technical support. Your computer has virus. Install AnyDesk now.", 1),
    ("Your loan has been approved. Pay processing fee via gift card.", 1),
    ("TRAI is cancelling your mobile number. Press 1 to speak to officer.", 1),
    # --- Safe (should be 0) ---
    ("Hi, how are you doing today?", 0),
    ("Can you please send me the meeting notes from yesterday?", 0),
    ("I wanted to confirm our dinner reservation for Saturday evening.", 0),
    ("The project deadline has been extended by two weeks.", 0),
    ("Please review the attached report and share your feedback.", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("The weather is great today. Let us go for a walk.", 0),
    ("Can you recommend a good restaurant near the office?", 0),
]


def run_classification_experiment(nlp_model, model_name):
    """Runs full precision/recall/F1 evaluation on synthetic test set."""
    print("\n" + "="*65)
    print("  EXPERIMENT 1 — Classification Metrics")
    print("="*65)

    y_true, y_pred = [], []
    latencies = []

    for text, label in SYNTHETIC_TEST_SET:
        t0 = time.perf_counter()
        is_scam, _, _ = classify_intent(text, nlp_model, model_name)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        y_true.append(label)
        y_pred.append(1 if is_scam else 0)

    # ── Metrics ──────────────────────────────────────────────────
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (tp + tn) / len(y_true)

    print(f"\n  Confusion Matrix:")
    print(f"  {'':10s}  Predicted SCAM  Predicted SAFE")
    print(f"  {'Actual SCAM':10s}  TP={tp:3d}            FN={fn:3d}")
    print(f"  {'Actual SAFE':10s}  FP={fp:3d}            TN={tn:3d}")
    print(f"\n  Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"\n  False Negatives (missed scams):")
    for (text, label), pred in zip(SYNTHETIC_TEST_SET, y_pred):
        if label == 1 and pred == 0:
            print(f"    ✗ \"{text}\"")
    print(f"\n  False Positives (wrongly flagged):")
    for (text, label), pred in zip(SYNTHETIC_TEST_SET, y_pred):
        if label == 0 and pred == 1:
            print(f"    ✗ \"{text}\"")

    return latencies, y_true, y_pred


def run_latency_experiment(nlp_model, model_name, n_runs=50):
    """
    Latency profiling — measures NLP inference time per chunk.
    Paper target: NLP inference < 45ms (Section VI-B).
    """
    print("\n" + "="*65)
    print("  EXPERIMENT 2 — Latency Profiling")
    print("="*65)

    test_texts = [t for t, _ in SYNTHETIC_TEST_SET]
    latencies = []
    for _ in range(n_runs):
        text = random.choice(test_texts)
        t0 = time.perf_counter()
        classify_intent(text, nlp_model, model_name)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(latencies)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mn  = np.min(latencies)
    mx  = np.max(latencies)

    print(f"\n  Runs      : {n_runs}")
    print(f"  Mean      : {avg:.2f} ms")
    print(f"  Min       : {mn:.2f} ms")
    print(f"  Max       : {mx:.2f} ms")
    print(f"  P95       : {p95:.2f} ms")
    print(f"  P99       : {p99:.2f} ms")
    print(f"\n  Paper target (NLP ≤ 45ms): "
          f"{'✓ MET' if avg <= 45 else '✗ NOT MET (CPU; use GPU/NPU for mobile)'}")
    print(f"  Real-time (<500ms total): "
          f"{'✓ VIABLE' if avg < 400 else '✗ Needs quantization'}")
    return latencies


def run_noise_robustness_experiment(nlp_model, model_name):
    """
    Simulates the noise injection described in Section V-B.
    Applies character-level corruption to mimic poor transcription/SNR.
    """
    print("\n" + "="*65)
    print("  EXPERIMENT 3 — Noise Robustness")
    print("="*65)

    scam_samples = [t for t, l in SYNTHETIC_TEST_SET if l == 1]

    def corrupt(text, level=0.10):
        """Randomly drop/swap characters to simulate ASR transcription noise."""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < level:
                action = random.choice(["drop", "swap"])
                if action == "drop":
                    chars[i] = ""
                else:
                    chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz ")
        return "".join(chars)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    print(f"\n  {'Noise Level':15s}  {'Recall':10s}  {'Example Corruption'}")
    print(f"  {'-'*60}")

    for noise in noise_levels:
        correct = 0
        example = ""
        for text in scam_samples:
            corrupted = corrupt(text, noise) if noise > 0 else text
            if not example:
                example = corrupted[:40]
            is_scam, _, _ = classify_intent(corrupted, nlp_model, model_name)
            if is_scam:
                correct += 1
        recall = correct / len(scam_samples)
        print(f"  {noise*100:>5.0f}%          {recall*100:>6.1f}%      \"{example}...\"")


def run_threshold_sensitivity(nlp_model, model_name):
    """
    Analyses how different τ_alert and τ_drop thresholds affect performance.
    Simulates temporal scoring over a synthetic scam call conversation.
    """
    print("\n" + "="*65)
    print("  EXPERIMENT 4 — Temporal Threshold Sensitivity")
    print("="*65)

    # Simulate a gradually escalating scam call (12 chunks)
    call_chunks = [
        ("Hello, is this Mr. Shaffan?", 0),
        ("I am calling from HDFC Bank customer care.", 0),
        ("We have noticed some suspicious activity on your account.", 1),
        ("Your account has been temporarily suspended.", 1),
        ("To restore access, we need to verify your identity.", 1),
        ("Can you please share your registered mobile number?", 1),
        ("Now, please confirm your account number.", 1),
        ("For security, we need your ATM PIN.", 1),
        ("And finally, the OTP you just received on your phone.", 1),
        ("Please hurry, the account will be permanently blocked in 5 minutes.", 1),
        ("Do not tell anyone about this call.", 1),
        ("Share the OTP now to save your account.", 1),
    ]

    tau_pairs = [(0.4, 0.7), (0.5, 0.75), (0.55, 0.80), (0.6, 0.85)]

    print(f"\n  {'τ_alert':10s} {'τ_drop':10s} {'Alert @chunk':15s} {'Drop @chunk':12s}")
    print(f"  {'-'*50}")

    for tau_a, tau_d in tau_pairs:
        scorer = TemporalThreatScorer(tau_alert=tau_a, tau_drop=tau_d)
        alert_at = drop_at = None

        for i, (text, _) in enumerate(call_chunks, 1):
            is_scam, conf, _ = classify_intent(text, nlp_model, model_name)
            chunk_score = conf if is_scam else (1.0 - conf)
            CT = scorer.update(chunk_score)
            if drop_at is None and CT >= tau_d:
                drop_at = i
            if alert_at is None and CT >= tau_a:
                alert_at = i

        print(f"  {tau_a:<10.2f} {tau_d:<10.2f} "
              f"{'chunk ' + str(alert_at) if alert_at else 'never':<15s} "
              f"{'chunk ' + str(drop_at) if drop_at else 'never':<12s}")


def run_all_experiments(nlp_model, model_name):
    latencies, y_true, y_pred = run_classification_experiment(nlp_model, model_name)
    lat_bench = run_latency_experiment(nlp_model, model_name)
    run_noise_robustness_experiment(nlp_model, model_name)
    run_threshold_sensitivity(nlp_model, model_name)

    # ── Optional: save latency histogram ─────────────────────────────────
    if MATPLOTLIB_OK:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Latency histogram
        axes[0].hist(lat_bench, bins=20, color="#2196F3", edgecolor="white")
        axes[0].axvline(np.mean(lat_bench), color="red", linestyle="--",
                        label=f"Mean={np.mean(lat_bench):.1f}ms")
        axes[0].set_title("NLP Inference Latency Distribution")
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        # Confusion matrix heatmap
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        cm = np.array([[tp, fn], [fp, tn]])
        im = axes[1].imshow(cm, cmap="Blues")
        axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Pred Scam", "Pred Safe"])
        axes[1].set_yticklabels(["Act Scam", "Act Safe"])
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=14, fontweight="bold")
        axes[1].set_title("Confusion Matrix")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        out_path = "scamguard_experiments.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n  📊 Experiment plots saved → {out_path}")
    else:
        print("\n  [INFO] Install matplotlib for plots: pip install matplotlib")


# =============================================================================
#  SECTION 8 — AUDIO FILE MODE  (uses Whisper ASR)
# =============================================================================

def run_audio_mode(audio_path: str, nlp_model, model_name):
    """
    Full pipeline: audio file → Whisper transcript → intent classification.
    Simulates real-time by splitting transcript into pseudo-chunks.
    """
    print(f"\n[AUDIO MODE] Processing: {audio_path}")
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return

    print(f"[TRANSCRIPT] {transcript}\n")

    # Split into ~sentence chunks
    import re
    sentences = re.split(r"[.!?,]\s*", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    scorer = TemporalThreatScorer()
    for i, chunk in enumerate(sentences, 1):
        is_scam, conf, details = classify_intent(chunk, nlp_model, model_name)
        chunk_score = conf if is_scam else (1.0 - conf)
        CT = scorer.update(chunk_score)
        status = scorer.status()
        print(f"  Chunk {i:02d}: [{status}] CT={CT:.3f} | \"{chunk}\"")
        if status == "DROP":
            print("  🚨  AUTO-DROP TRIGGERED — Malicious call detected.")
            break


# =============================================================================
#  MAIN
# =============================================================================

def main():
    import sys
    print("\n" + "="*65)
    print("  🛡  SCAM GUARD — Enhanced Hybrid AI Engine v2.0  🛡")
    print("="*65)
    print("\n  Loading NLP model...")
    nlp_model, model_name = load_nlp_model()

    # Parse command-line args
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
            print("    python scamguard_enhanced.py experiments  # Run all paper experiments")
            print("    python scamguard_enhanced.py audio <path> # Transcribe+classify audio")
            return

    # Default: interactive CLI
    run_interactive_mode(nlp_model, model_name)


if __name__ == "__main__":
    main()