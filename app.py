"""HarvestAI Flask app. Load Kaggle MobileNetV2 .pth checkpoint for REAL predictions."""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT = MODELS_DIR / "best_mobilenetv2_full.pth"
METRICS = MODELS_DIR / "evaluation.json"
CLASSES = ("UNRIPE", "RIPE", "OVERRIPE")
MAX_BYTES = 6 * 1024 * 1024

SUPPORTED_FRUITS = {"banana", "orange", "apple"}
REJECT_NON_PRODUCE = {
    "person", "laptop", "keyboard", "mouse", "cell phone", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "chair", "couch",
    "bed", "toilet", "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "scissors", "teddy bear", "hair drier", "toothbrush"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

preprocess = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
verifier_transform = transforms.ToTensor()

_MODEL = None
_MODEL_ERROR = None
_VERIFIER = None
_VERIFIER_ERROR = None


def load_model():
    """Initialize on first request, then cache. Do not use mock weights."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if not CHECKPOINT.is_file():
        _MODEL_ERROR = ("Trained model not found. Download best_mobilenetv2_full.pth "
                        "from Kaggle and place it in the models folder.")
        raise FileNotFoundError(_MODEL_ERROR)
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        # This project loads locally owned files, not checkpoints from strangers.
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        model.eval()
        _MODEL = model
        _MODEL_ERROR = None
        return _MODEL
    except Exception as exc:
        _MODEL_ERROR = f"Could not load Kaggle model: {type(exc).__name__}: {exc}"
        raise RuntimeError(_MODEL_ERROR) from exc


def load_verifier():
    """Initialize pretrained SSDLite320 detector for fruit verification."""
    global _VERIFIER, _VERIFIER_ERROR
    if _VERIFIER is not None:
        return _VERIFIER
    try:
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        verifier = ssdlite320_mobilenet_v3_large(weights=weights)
        verifier.eval()
        _VERIFIER = (verifier, weights.meta["categories"])
        _VERIFIER_ERROR = None
        return _VERIFIER
    except Exception as exc:
        _VERIFIER_ERROR = f"Could not load verification model: {type(exc).__name__}: {exc}"
        raise RuntimeError(_VERIFIER_ERROR) from exc


def verify_fruit(image: Image.Image) -> tuple[bool, dict]:
    """Verify if the image contains a supported fruit using real object detection.
    
    Rejects unrelated objects (people, laptops, keyboards, cars, etc.) and images
    where supported fruit cannot be confidently established.
    """
    verifier, categories = load_verifier()
    tensor = verifier_transform(image).unsqueeze(0)
    with torch.inference_mode():
        preds = verifier(tensor)[0]

    scores = preds["scores"].tolist()
    labels = [categories[idx] for idx in preds["labels"].tolist()]

    all_detections = [
        {"label": l, "score": round(s, 3)}
        for l, s in zip(labels, scores)
        if s >= 0.20
    ]

    fruit_detections = [
        d for d in all_detections
        if d["label"] in SUPPORTED_FRUITS and d["score"] >= 0.35
    ]
    reject_detections = [
        d for d in all_detections
        if d["label"] in REJECT_NON_PRODUCE and d["score"] >= 0.40
    ]

    if not fruit_detections:
        return False, {
            "reason": "No supported fruit detected",
            "detected_objects": all_detections[:5]
        }

    best_fruit = max(fruit_detections, key=lambda d: d["score"])

    # If a prominent non-produce object (e.g. person, keyboard) is detected with higher confidence
    if reject_detections:
        best_reject = max(reject_detections, key=lambda d: d["score"])
        if best_reject["score"] > best_fruit["score"] and best_fruit["score"] < 0.60:
            return False, {
                "reason": f"Prominent non-fruit object ({best_reject['label']}) detected",
                "detected_objects": all_detections[:5]
            }

    return True, {
        "detected_fruit": best_fruit["label"],
        "detection_confidence": round(best_fruit["score"] * 100, 2),
        "detected_objects": all_detections[:5]
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        load_model()
        load_verifier()
        return jsonify(
            status="ready",
            model_loaded=True,
            verifier_loaded=True,
            supported_fruits=sorted(list(SUPPORTED_FRUITS))
        )
    except (FileNotFoundError, RuntimeError) as exc:
        return jsonify(status="model_missing_or_invalid", model_loaded=False, message=str(exc)), 503


@app.get("/api/metrics")
@app.get("/api/evaluation")
def metrics():
    if not METRICS.is_file():
        return jsonify(error="Run Kaggle evaluation and copy evaluation.json to models/"), 404
    try:
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        cm = data.get("confusion_matrix", [])
        class_names = data.get("class_names", ["Unripe", "Ripen", "Overripe"])
        if len(cm) == 3 and "per_class" not in data:
            per_class = []
            for i, name in enumerate(class_names):
                tp = cm[i][i]
                support = sum(cm[i])
                pred_count = sum(cm[r][i] for r in range(len(cm)))
                prec = (tp / pred_count * 100) if pred_count > 0 else 0.0
                rec = (tp / support * 100) if support > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                per_class.append({
                    "class": name,
                    "precision_pct": round(prec, 4),
                    "recall_pct": round(rec, 4),
                    "f1_pct": round(f1, 4),
                    "support": int(support),
                    "predicted_count": int(pred_count)
                })
            data["per_class"] = per_class
        if "accuracy_pct" not in data and "accuracy" in data:
            data["accuracy_pct"] = data["accuracy"]
        if "macro_f1_pct" not in data and "macro_f1" in data:
            data["macro_f1_pct"] = data["macro_f1"]
        if "macro_precision_pct" not in data and "macro_precision" in data:
            data["macro_precision_pct"] = data["macro_precision"]
        if "macro_recall_pct" not in data and "macro_recall" in data:
            data["macro_recall_pct"] = data["macro_recall"]
        return jsonify(data)
    except (ValueError, OSError):
        return jsonify(error="evaluation.json cannot be read"), 500


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image uploaded. Send multipart field named 'image'."), 400
    upload = request.files["image"]
    if not upload.filename:
        return jsonify(error="Choose an image first."), 400
    if upload.mimetype not in ("image/jpeg", "image/png", "image/webp"):
        return jsonify(error="Only JPEG, PNG, and WebP images are accepted."), 400
    try:
        # No file is saved permanently; images are decoded in memory.
        binary = upload.read(MAX_BYTES + 1)
        if len(binary) > MAX_BYTES:
            return jsonify(error="Maximum image size is 6 MB."), 413
        with Image.open(BytesIO(binary)) as source:
            source.verify()
        with Image.open(BytesIO(binary)) as source:
            if source.width * source.height > 20_000_000:
                return jsonify(error="Image resolution is too large."), 400
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="Cannot read this image. Try a standard JPEG or PNG."), 400

    # Verification stage: separate pretrained object detection model
    try:
        load_verifier()
    except Exception as exc:
        return jsonify(error=f"Verification model unavailable: {exc}"), 503

    verified, v_info = verify_fruit(image)
    if not verified:
        return jsonify({
            "status": "unsupported_or_uncertain_image",
            "verified": False,
           "message": "Unwanted object detected! Please upload a clear image of a supported fruit (apple, banana, or orange).",
            "detected_objects": v_info.get("detected_objects", []),
            "supported_fruits": sorted(list(SUPPORTED_FRUITS)),
            "model_filter": "SSDLite320 MobileNetV3 (COCO Object Detection)",
            "limitations": "Supports COCO fruit classes: banana, orange, apple. Non-produce and unverified images are safely rejected."
        }), 200

    # Maturity classifier runs ONLY after fruit verification succeeds
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 503

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
    winner = max(range(3), key=probabilities.__getitem__)
    confidence = round(100 * probabilities[winner], 2)
    return jsonify({
        "status": "success",
        "verified": True,
        "detected_fruit": v_info.get("detected_fruit"),
        "detection_confidence": v_info.get("detection_confidence"),
        "class": CLASSES[winner],
        "confidence": confidence,
        "is_uncertain": confidence < 70.0,
        "all_predictions": [
            {"class": name, "confidence": round(100 * value, 2)}
            for name, value in zip(CLASSES, probabilities)
        ],
        "model": "MobileNetV2 (PyTorch)",
        "verification_model": "SSDLite320 MobileNetV3 (COCO Object Detection)",
        "note": "Confidence is a model score, not a guarantee of correct maturity.",
    })


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Maximum image upload size is 6 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
