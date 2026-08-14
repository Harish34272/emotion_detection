def section(title):
    print(f"\n--- {title} ---")

def run(label, fn):
    try:
        result = fn()
        print(f"✅ {label}: {result if result is not None else 'ok'}")
        return True
    except ImportError as e:
        print(f"❌ {label} failed (ImportError): {e}")
    except Exception as e:
        print(f"⚠️  {label} failed ({type(e).__name__}): {e}")
    return False

results = {}

section("cv2")
def check_cv2():
    import cv2
    return cv2.__version__
results['cv2'] = run("cv2 version", check_cv2)

section("onnxruntime")
def check_onnx():
    import onnxruntime as ort
    return f"{ort.__version__} providers={ort.get_available_providers()}"
results['onnxruntime'] = run("onnxruntime", check_onnx)

section("insightface FaceAnalysis")
def check_insightface():
    from insightface.app import FaceAnalysis
    a = FaceAnalysis(providers=['CPUExecutionProvider'])
    a.prepare(ctx_id=-1, det_size=(320, 320))
    return "initialized"
results['insightface'] = run("insightface", check_insightface)

section("deepface")
def check_deepface():
    from deepface import DeepFace
    return "imported"
results['deepface'] = run("deepface", check_deepface)

section("deepface + insightface combined")
def check_combined():
    from deepface import DeepFace
    from insightface.app import FaceAnalysis
    return "both imported"
results['combined'] = run("deepface+insightface", check_combined)

section("Summary")
for name, ok in results.items():
    print(f"{name}: {'OK' if ok else 'FAILED'}")

if all(results.values()):
    print("\nAll checks passed.")
else:
    failed = [n for n, ok in results.items() if not ok]
    print(f"\nFailed checks: {', '.join(failed)}")
