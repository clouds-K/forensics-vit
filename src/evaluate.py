"""
evaluate.py
-----------
Evaluation complète du ForensicsViT avec :
  - Métriques finales (accuracy, F1, AUC)
  - Matrice de confusion
  - Visualisation Grad-CAM (zones suspectes surlignées)
  - Rapport forensique automatique par image

Usage :
  python src/evaluate.py                        <- évalue sur le test set
  python src/evaluate.py --image <path>         <- analyse une image unique
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders, get_transforms, get_ela_transforms, IMAGE_SIZE
from model import create_model
from ela_generator import generate_ela, compute_sha256

BASE = Path(__file__).parent.parent


# ─────────────────────────────────────────
#  GRAD-CAM pour ViT
# ─────────────────────────────────────────

class ViTGradCAM:
    """
    Grad-CAM adapté pour Vision Transformer.
    Utilise les gradients du dernier bloc d'attention pour
    localiser les zones qui ont influencé la décision.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Accroche les hooks sur le dernier bloc MLP du ViT."""
        target_layer = self.model.vit.blocks[-1].mlp

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor_6ch: torch.Tensor, target_class: int = 1):
        """
        Génère la heatmap Grad-CAM.

        Args:
            tensor_6ch   : tensor (1, 6, 224, 224)
            target_class : 1 = tampered (classe à expliquer)

        Returns:
            heatmap : numpy array (224, 224) normalisé [0, 1]
        """
        self.model.eval()
        tensor_6ch = tensor_6ch.requires_grad_(True)

        # Forward pass
        logits = self.model(tensor_6ch)
        score  = logits[0, target_class]

        # Backward
        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        # Grad-CAM : moyenne pondérée des activations par les gradients
        # Shape : (seq_len, hidden_dim) — on ignore le CLS token (idx 0)
        grads = self.gradients[0, 1:, :]       # (196, 3072)
        acts  = self.activations[0, 1:, :]     # (196, 3072)

        weights = grads.mean(dim=-1)           # (196,)
        cam = (weights.unsqueeze(-1) * acts).sum(dim=-1)  # (196,)
        cam = torch.relu(cam)

        # Reshape : 196 patches → 14×14 (pour image 224/16=14)
        n_patches = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(n_patches, n_patches).numpy()

        # Upscale vers 224×224
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalisation [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ─────────────────────────────────────────
#  ANALYSE D'UNE IMAGE UNIQUE
# ─────────────────────────────────────────

def analyze_single_image(image_path: str, model: nn.Module, device: torch.device):
    """
    Analyse forensique complète d'une image unique.
    Produit : ELA + Grad-CAM + rapport forensique.
    """
    img_path = Path(image_path)
    print(f"\n  Analyse forensique : {img_path.name}")

    # ── Hash SHA-256 (chaîne de custody) ───────────────────────
    sha256 = compute_sha256(str(img_path))
    print(f"  SHA-256 : {sha256[:48]}...")

    # ── Génération ELA ──────────────────────────────────────────
    original  = Image.open(img_path).convert('RGB')
    ela_image, max_diff = generate_ela(str(img_path))
    print(f"  ELA max_diff : {max_diff:.1f}/255")

    # ── Préparation tensor 6 canaux ─────────────────────────────
    tf_rgb = get_transforms("val")
    tf_ela = get_ela_transforms("val")

    t_rgb = tf_rgb(original).unsqueeze(0)   # (1, 3, 224, 224)
    t_ela = tf_ela(ela_image).unsqueeze(0)  # (1, 3, 224, 224)
    tensor = torch.cat([t_rgb, t_ela], dim=1).to(device)  # (1, 6, 224, 224)

    # ── Inférence ───────────────────────────────────────────────
    model.eval()
    gradcam = ViTGradCAM(model)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    prob_authentic = probs[0].item()
    prob_tampered  = probs[1].item()
    prediction     = "TAMPERED" if prob_tampered > 0.5 else "AUTHENTIC"

    print(f"  Prediction   : {prediction}")
    print(f"  Confiance    : authentic={prob_authentic:.3f} | tampered={prob_tampered:.3f}")

    # ── Grad-CAM ────────────────────────────────────────────────
    # Re-run avec gradients pour Grad-CAM
    tensor_grad = torch.cat([t_rgb, t_ela], dim=1).to(device)
    heatmap = gradcam.generate(tensor_grad, target_class=1)

    # ── Visualisation ───────────────────────────────────────────
    orig_np = np.array(original.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    ela_np  = np.array(ela_image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    # Superposition heatmap sur image originale
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * orig_np + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    # Figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    color = 'red' if prediction == "TAMPERED" else 'green'

    fig.suptitle(
        f"Rapport Forensique — {img_path.name}\n"
        f"Verdict : {prediction}  |  "
        f"P(authentic)={prob_authentic:.3f}  P(tampered)={prob_tampered:.3f}\n"
        f"SHA-256 : {sha256[:32]}...",
        fontsize=9, color=color, fontweight='bold'
    )

    axes[0].imshow(orig_np)
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    axes[1].imshow(ela_np, cmap='hot')
    axes[1].set_title(f"Carte ELA\n(max_diff={max_diff:.0f})")
    axes[1].axis('off')

    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title("Grad-CAM\n(zones decisives)")
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("Superposition\n(rouge=suspect)")
    axes[3].axis('off')

    plt.tight_layout()

    # Sauvegarde
    out_path = BASE / "reports" / f"forensic_{img_path.stem}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"  Rapport visuel : {out_path}")
    plt.show()
    plt.close()

    # ── Rapport JSON ────────────────────────────────────────────
    report = {
        "filename"      : img_path.name,
        "sha256"        : sha256,
        "prediction"    : prediction,
        "prob_authentic": round(prob_authentic, 4),
        "prob_tampered" : round(prob_tampered, 4),
        "ela_max_diff"  : round(max_diff, 1),
        "verdict"       : "FALSIFIEE" if prediction == "TAMPERED" else "AUTHENTIQUE",
        "confiance"     : round(max(prob_authentic, prob_tampered) * 100, 1),
    }

    json_path = BASE / "reports" / f"forensic_{img_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Rapport JSON   : {json_path}")

    return report


# ─────────────────────────────────────────
#  ÉVALUATION SUR LE TEST SET
# ─────────────────────────────────────────

def evaluate_test_set(model, test_loader, device):
    """Métriques complètes + matrice de confusion + courbe ROC."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for tensors, labels, _ in test_loader:
            tensors = tensors.to(device)
            logits  = model(tensors)
            probs   = torch.softmax(logits, dim=1)[:, 1]
            preds   = logits.argmax(dim=1)

            all_preds  += preds.cpu().tolist()
            all_labels += labels.tolist()
            all_probs  += probs.cpu().tolist()

    # ── Métriques ───────────────────────────────────────────────
    print("\n  Rapport de classification :")
    report = classification_report(
        all_labels, all_preds,
        target_names=["authentic", "tampered"]
    )
    print(report)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"  AUC-ROC : {auc:.4f}")
    except Exception:
        auc = None
        print("  AUC-ROC : non calculable (classes manquantes dans le batch)")

    # ── Figure : Confusion Matrix + ROC ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation ForensicsViT — Test Set", fontsize=13)

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["authentic", "tampered"]
    )
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title("Matrice de Confusion")

    # Courbe ROC
    if auc is not None:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange', lw=2)
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlabel("Faux positifs (FPR)")
        axes[1].set_ylabel("Vrais positifs (TPR)")
        axes[1].set_title("Courbe ROC")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "AUC non disponible",
                     ha='center', va='center', fontsize=12)

    plt.tight_layout()
    out = BASE / "reports" / "evaluation_results.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"\n  Figure sauvegardee : {out}")
    plt.show()
    plt.close()

    return all_labels, all_preds, all_probs, auc


# ─────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")

    # ── Chargement du modèle ────────────────────────────────────
    ckpt_path = BASE / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  ERREUR : {ckpt_path} introuvable.")
        print("  Lance d'abord : python src/train.py")
        sys.exit(1)

    print("=" * 55)
    print("  Chargement du meilleur modele...")
    print("=" * 55)

    model = create_model(pretrained=False)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    print(f"  Modele charge (epoch {checkpoint['epoch']})")

    # ── Mode image unique ───────────────────────────────────────
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            image_path = sys.argv[idx + 1]
            analyze_single_image(image_path, model, device)
        else:
            print("  Usage : python src/evaluate.py --image <chemin_image>")
        sys.exit(0)

    # ── Mode évaluation complète ────────────────────────────────
    print("\nChargement du dataset de test...")
    _, _, test_loader, _ = create_dataloaders(
        raw_dir     = str(BASE / "data" / "raw"),
        ela_dir     = str(BASE / "data" / "ela"),
        batch_size  = 8,
        max_samples = 500,
        num_workers = 0
    )

    print("\n" + "=" * 55)
    print("  EVALUATION SUR LE TEST SET")
    print("=" * 55)
    labels, preds, probs, auc = evaluate_test_set(model, test_loader, device)

    # ── Demo Grad-CAM sur quelques images ───────────────────────
    print("\n" + "=" * 55)
    print("  DEMO GRAD-CAM (3 images tampered)")
    print("=" * 55)

    tampered_dir = BASE / "data" / "raw" / "tampered"
    tampered_imgs = sorted(tampered_dir.glob("*.jpg"))[:3]

    for img_path in tampered_imgs:
        analyze_single_image(str(img_path), model, device)

    print("\n" + "=" * 55)
    print("  Evaluation complete !")
    print(f"  Rapports dans : {BASE / 'reports'}")
    print("=" * 55)