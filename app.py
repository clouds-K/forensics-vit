"""
app.py
------
Interface de démonstration interactive du ForensicsViT.
Lance un serveur web local avec une interface pour analyser des images.

Usage :
  python app.py
  → Ouvre http://localhost:5000 dans ton navigateur
"""

import sys
import json
import base64
import hashlib
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')  # pas de GUI, render en mémoire
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import create_model
from ela_generator import generate_ela, compute_sha256
from dataset import get_transforms, get_ela_transforms, IMAGE_SIZE
from evaluate import ViTGradCAM

# ─────────────────────────────────────────
BASE   = Path(__file__).parent
CKPT   = BASE / "checkpoints" / "best_model.pt"
DEVICE = torch.device("cpu")

app = Flask(__name__, static_folder="demo_ui", static_url_path="")

# Chargement du modèle au démarrage
print("Chargement du modèle...")
model = create_model(pretrained=False)
if CKPT.exists():
    ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Modèle chargé (epoch {ckpt['epoch']})")
else:
    print("  ATTENTION : best_model.pt introuvable, poids aléatoires")
model.to(DEVICE)
model.eval()

gradcam = ViTGradCAM(model)


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def numpy_to_base64(arr: np.ndarray) -> str:
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    return pil_to_base64(img)


@app.route("/")
def index():
    return send_from_directory("demo_ui", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Endpoint principal : reçoit une image, retourne l'analyse forensique."""
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # ── Hash SHA-256 ───────────────────────────────────────────
    sha256 = hashlib.sha256(img_bytes).hexdigest()

    # ── Chargement image ───────────────────────────────────────
    original = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Sauvegarde temporaire pour ELA
    tmp = BASE / "reports" / "_tmp_upload.jpg"
    original.save(str(tmp), "JPEG", quality=95)

    # ── ELA ────────────────────────────────────────────────────
    ela_image, max_diff = generate_ela(str(tmp))
    tmp.unlink(missing_ok=True)

    # ── Tensor 6 canaux ────────────────────────────────────────
    tf_rgb = get_transforms("val")
    tf_ela = get_ela_transforms("val")
    t_rgb  = tf_rgb(original).unsqueeze(0)
    t_ela  = tf_ela(ela_image).unsqueeze(0)
    tensor = torch.cat([t_rgb, t_ela], dim=1)

    # ── Inférence ──────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    prob_auth = probs[0].item()
    prob_tamp = probs[1].item()
    prediction = "TAMPERED" if prob_tamp > 0.5 else "AUTHENTIC"

    # ── Grad-CAM ───────────────────────────────────────────────
    tensor_grad = torch.cat([t_rgb, t_ela], dim=1)
    heatmap = gradcam.generate(tensor_grad, target_class=1)

    # Images pour l'interface
    orig_resized = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    orig_np = np.array(orig_resized) / 255.0
    ela_np  = np.array(ela_image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    # Overlay Grad-CAM
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.6 * orig_np + 0.4 * heatmap_color, 0, 1)

    # Heatmap ELA en colormap
    ela_gray = np.array(ela_image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
    ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)
    ela_colored = cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB)

    return jsonify({
        "prediction"    : prediction,
        "prob_authentic": round(prob_auth * 100, 1),
        "prob_tampered" : round(prob_tamp * 100, 1),
        "ela_max_diff"  : round(max_diff, 1),
        "sha256"        : sha256,
        "filename"      : file.filename,
        "images": {
            "original" : numpy_to_base64(orig_np),
            "ela"      : pil_to_base64(Image.fromarray(ela_colored)),
            "gradcam"  : numpy_to_base64(
                plt.cm.jet(heatmap)[:, :, :3]
            ),
            "overlay"  : numpy_to_base64(overlay),
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ForensicsViT — Demo Interactive")
    print("=" * 50)
    print("  Ouvre ton navigateur sur : http://localhost:5000")
    print("  Ctrl+C pour arrêter\n")
    app.run(debug=False, host="0.0.0.0", port=5000)