"""Test all v5.0 API endpoints."""
import urllib.request
import json

BASE = "http://localhost:8000"

def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(r.read())

def post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

print("=" * 60)
print("  SCAMGUARD v5.0 — API ENDPOINT TESTS")
print("=" * 60)

# 1. Health
print("\n[1] GET /api/v1/health")
h = get("/api/v1/health")
print(f"    Status: {h['status']} | Model: {h['model']}")
print(f"    Inference OK: {h.get('inference_ok')} | Latency: {h.get('inference_ms')}ms")
print(f"    Uptime: {h['uptime_seconds']}s | Build: {h['build']}")

# 2. Version
print("\n[2] GET /api/v1/version")
v = get("/api/v1/version")
print(f"    {v['name']} v{v['version']} (build: {v['build']})")

# 3. Keywords
print("\n[3] GET /api/v1/keywords")
kw = get("/api/v1/keywords")
for tier, data in kw.items():
    words = data["words"][:5]
    print(f"    {tier} (weight={data['weight']}): {', '.join(words)}...")

# 4. Single Analysis
print("\n[4] POST /api/v1/analyze (safe text)")
r1 = post("/api/v1/analyze", {"text": "Hey bhai, kal office kitne baje aana hai?"})
print(f"    SCAM={r1['is_scam']} | CT={r1['ct_percent']}% | Status={r1['status']}")

print("\n[5] POST /api/v1/analyze (scam text)")
r2 = post("/api/v1/analyze", {"text": "Your account is blocked. Share your OTP and Aadhaar immediately or face arrest."})
print(f"    SCAM={r2['is_scam']} | CT={r2['ct_percent']}% | Status={r2['status']}")
print(f"    Keywords: {[k['word'] for k in r2['matched_keywords']]}")
print(f"    Explanation: {r2['explanation']}")

# 5. Batch Analysis
print("\n[6] POST /api/v1/analyze-batch")
batch = post("/api/v1/analyze-batch", {
    "texts": [
        "Happy birthday! Wishing you all the best.",
        "Install AnyDesk and share the code for refund.",
        "Meeting scheduled for tomorrow at 10am."
    ]
})
print(f"    Batch: {batch['count']} texts analyzed")
for r in batch["results"]:
    label = "SCAM" if r["is_scam"] else "SAFE"
    print(f"    [{label}] CT={r['ct_percent']:.1f}% | {r['text'][:50]}")

# 6. Feedback
print("\n[7] POST /api/v1/feedback")
fb = post("/api/v1/feedback", {
    "session_id": "test-session",
    "chunk_text": "Test feedback",
    "predicted_scam": True,
    "actual_scam": False,
    "comment": "False positive"
})
print(f"    {fb['status']}: {fb['message']}")

# 7. Metrics
print("\n[8] GET /api/v1/metrics")
m = get("/api/v1/metrics")
print(f"    Total analyses: {m['total_analyses']}")
print(f"    Scams detected: {m['total_scams_detected']}")
print(f"    Avg latency: {m['avg_latency_ms']:.1f}ms")
print(f"    Active sessions: {m['active_sessions']}")

# 8. Legacy endpoint compatibility
print("\n[9] Legacy endpoint: GET /api/health")
h2 = get("/api/health")
print(f"    Status: {h2['status']} (backward compatible)")

print("\n" + "=" * 60)
print("  ALL ENDPOINTS VERIFIED SUCCESSFULLY!")
print("=" * 60)
