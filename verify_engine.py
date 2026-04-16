"""Quick verification script for ScamGuard v5.0 engine."""
from scamguard_enhanced import classify_intent, load_nlp_model

print("Loading model...")
nlp, name = load_nlp_model()
print(f"Model: {name}\n")

tests = [
    ("Hey bhai, kal office kitne baje aana hai?", False),
    ("Share your OTP immediately or your account will be blocked", True),
    ("Never share your OTP with anyone", False),
    ("Sir main SBI branch se bol raha hu, aapka khata band hone wala hai", True),
    ("Can you send me the meeting agenda?", False),
    ("This is police. Transfer money to avoid arrest.", True),
    ("Aakhri mauka! Download anydesk right now to claim your refund.", True),
    ("Happy birthday! Wishing you all the best.", False),
]

correct = 0
for text, expected_scam in tests:
    is_scam, conf, details = classify_intent(text, nlp, name)
    ok = is_scam == expected_scam
    correct += int(ok)
    mark = "OK" if ok else "X "
    label = "SCAM" if is_scam else "SAFE"
    print(f"  [{mark}] [{label}] conf={conf:.3f} kw={details['keyword_weight']} | {text[:65]}")

print(f"\nResult: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")
print("Engine verification complete!")
