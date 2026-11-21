"""
train.py
--------
Entraînement du ForensicsViT en 2 phases optimisé pour CPU.

Phase 1 : backbone gelé  → entraîne seulement le classifier (rapide)
Phase 2 : fine-tuning    → tout le modèle avec lr très faible

Optimisations CPU :
  - batch_size réduit (8)
  - max_samples pour limiter le dataset en dev
  - gradient accumulation pour simuler un grand batch
  - sauvegarde du meilleur modèle automatique

Usage :
  python src/train.py              ← entraînement rapide (500 images, 5 epochs)
  python src/train.py --full       ← entraînement complet (tout le dataset)
"""

import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from model import create_model


# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

BASE = Path(__file__).parent.parent

CONFIG = {
    # Données
    "raw_dir"     : str(BASE / "data" / "raw"),
    "ela_dir"     : str(BASE / "data" / "ela"),

    # Entraînement rapide (mode dev, CPU)
    "max_samples" : 500,      # None = dataset complet
    "batch_size"  : 8,
    "accum_steps" : 4,        # gradient accumulation (batch effectif = 32)

    # Phase 1 : classifier seulement
    "phase1_epochs" : 3,
    "phase1_lr"     : 1e-3,

    # Phase 2 : fine-tuning complet
    "phase2_epochs" : 5,
    "phase2_lr"     : 1e-5,

    # Sauvegarde
    "checkpoint_dir": str(BASE / "checkpoints"),
    "results_dir"   : str(BASE / "reports"),
}


# ─────────────────────────────────────────
#  MÉTRIQUES
# ─────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    """Calcule accuracy, F1, et le rapport sklearn."""
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    report = classification_report(
        all_labels, all_preds,
        target_names=["authentic", "tampered"],
        output_dict=True,
        zero_division=0
    )
    f1_tampered = report["tampered"]["f1-score"]
    f1_macro    = report["macro avg"]["f1-score"]

    return {
        "accuracy"    : accuracy,
        "f1_tampered" : f1_tampered,
        "f1_macro"    : f1_macro,
        "report"      : report
    }


# ─────────────────────────────────────────
#  BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps=1):
    """Une epoch d'entraînement avec gradient accumulation."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()

    for step, (tensors, labels, _) in enumerate(loader):
        tensors = tensors.to(device)
        labels  = labels.to(device)

        # Forward
        logits = model(tensors)
        loss   = criterion(logits, labels) / accum_steps

        # Backward
        loss.backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds  += preds
        all_labels += labels.cpu().tolist()

        # Progression
        if (step + 1) % 10 == 0:
            acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            print(f"    step {step+1:3d}/{len(loader)} | "
                  f"loss={total_loss/(step+1):.4f} | acc={acc:.3f}", end='\r')

    print()
    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds, [])
    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    """Évaluation sur val ou test."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for tensors, labels, _ in loader:
            tensors = tensors.to(device)
            labels  = labels.to(device)

            logits = model(tensors)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds  = logits.argmax(dim=1).cpu().tolist()

            all_preds  += preds
            all_labels += labels.cpu().tolist()
            all_probs  += probs

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


# ─────────────────────────────────────────
#  SAUVEGARDE / CHARGEMENT
# ─────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    """Sauvegarde le modèle + métadonnées."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "metrics"   : metrics,
    }, path)
    tag = " [MEILLEUR]" if is_best else ""
    print(f"  Checkpoint sauvegardé : {Path(path).name}{tag}")


# ─────────────────────────────────────────
#  COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────

def plot_training_curves(history, output_path):
    """Génère et sauvegarde les courbes loss / accuracy / F1."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Courbes d'apprentissage — ForensicsViT", fontsize=13)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", marker='o')
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker='s')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", marker='o')
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker='s')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 tampered
    axes[2].plot(epochs, history["val_f1_tampered"], label="F1 tampered", marker='s', color='red')
    axes[2].plot(epochs, history["val_f1_macro"],    label="F1 macro",    marker='o', color='purple')
    axes[2].set_title("F1-Score (classe falsifiee)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Courbes sauvegardees : {output_path}")
    plt.close()


# ─────────────────────────────────────────
#  ENTRAÎNEMENT PRINCIPAL
# ─────────────────────────────────────────

def train(config: dict):
    """Pipeline d'entraînement complet en 2 phases."""

    device = torch.device("cpu")
    print(f"\n  Device : {device}")

    # ── Données ─────────────────────────────────────────────────
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        raw_dir     = config["raw_dir"],
        ela_dir     = config["ela_dir"],
        batch_size  = config["batch_size"],
        max_samples = config["max_samples"],
        num_workers = 0
    )

    # ── Modèle ──────────────────────────────────────────────────
    print("\nInitialisation du modele...")
    model = create_model(pretrained=True).to(device)

    # Loss avec poids de classe (dataset déséquilibré)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Historique pour les courbes
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : [],
        "val_f1_tampered": [], "val_f1_macro": []
    }

    best_val_f1  = 0.0
    ckpt_dir     = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt    = ckpt_dir / "best_model.pt"
    last_ckpt    = ckpt_dir / "last_model.pt"

    total_start = time.time()

    # ════════════════════════════════════════════════════════════
    #  PHASE 1 : Backbone gelé — entraîne seulement le classifier
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  PHASE 1 : Classifier seulement (backbone gele)")
    print("=" * 55)

    model.freeze_backbone(freeze=True)
    trainable, total = model.count_parameters()
    print(f"  Params entraînables : {trainable:,} / {total:,}")

    optimizer1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["phase1_lr"],
        weight_decay=1e-4
    )
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=config["phase1_epochs"])

    for epoch in range(1, config["phase1_epochs"] + 1):
        t0 = time.time()
        print(f"\n  [Phase 1 | Epoch {epoch}/{config['phase1_epochs']}]")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer1, criterion, device,
            accum_steps=config["accum_steps"]
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler1.step()

        elapsed = time.time() - t0
        print(f"  Train : loss={train_loss:.4f} | acc={train_metrics['accuracy']:.3f}")
        print(f"  Val   : loss={val_loss:.4f}   | acc={val_metrics['accuracy']:.3f} "
              f"| F1_tampered={val_metrics['f1_tampered']:.3f} "
              f"| F1_macro={val_metrics['f1_macro']:.3f}")
        print(f"  Temps : {elapsed:.1f}s")

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_tampered"].append(val_metrics["f1_tampered"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        # Meilleur modèle
        is_best = val_metrics["f1_macro"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, optimizer1, epoch, val_metrics, str(best_ckpt), is_best=True)

    # ════════════════════════════════════════════════════════════
    #  PHASE 2 : Fine-tuning complet
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  PHASE 2 : Fine-tuning complet (lr tres faible)")
    print("=" * 55)

    model.freeze_backbone(freeze=False)
    trainable, total = model.count_parameters()
    print(f"  Params entraînables : {trainable:,} / {total:,}")

    optimizer2 = AdamW(
        model.parameters(),
        lr=config["phase2_lr"],
        weight_decay=1e-4
    )
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=config["phase2_epochs"])

    for epoch in range(1, config["phase2_epochs"] + 1):
        t0 = time.time()
        print(f"\n  [Phase 2 | Epoch {epoch}/{config['phase2_epochs']}]")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer2, criterion, device,
            accum_steps=config["accum_steps"]
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler2.step()

        elapsed = time.time() - t0
        print(f"  Train : loss={train_loss:.4f} | acc={train_metrics['accuracy']:.3f}")
        print(f"  Val   : loss={val_loss:.4f}   | acc={val_metrics['accuracy']:.3f} "
              f"| F1_tampered={val_metrics['f1_tampered']:.3f} "
              f"| F1_macro={val_metrics['f1_macro']:.3f}")
        print(f"  Temps : {elapsed:.1f}s")

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_tampered"].append(val_metrics["f1_tampered"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        # Meilleur modèle
        is_best = val_metrics["f1_macro"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, optimizer2, epoch, val_metrics, str(best_ckpt), is_best=True)

    # Dernier checkpoint
    save_checkpoint(model, optimizer2, epoch, val_metrics, str(last_ckpt))

    # ── Évaluation finale sur le test set ───────────────────────
    print("\n" + "=" * 55)
    print("  EVALUATION FINALE SUR LE TEST SET")
    print("=" * 55)

    # Charge le meilleur modèle
    checkpoint = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  F1 tampered   : {test_metrics['f1_tampered']:.4f}")
    print(f"  F1 macro      : {test_metrics['f1_macro']:.4f}")

    print("\n  Rapport de classification :")
    report = test_metrics["report"]
    for cls in ["authentic", "tampered"]:
        r = report[cls]
        print(f"    {cls:12s} : precision={r['precision']:.3f} "
              f"recall={r['recall']:.3f} f1={r['f1-score']:.3f}")

    # ── Courbes et sauvegarde ─────────────────────────────────
    curves_path = Path(config["results_dir"]) / "training_curves.png"
    plot_training_curves(history, str(curves_path))

    # Sauvegarde JSON des résultats
    results = {
        "config"      : {k: v for k, v in config.items() if k not in ["raw_dir", "ela_dir"]},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"},
        "best_val_f1" : best_val_f1,
        "total_time_s": round(time.time() - total_start, 1),
        "history"     : history
    }
    results_json = Path(config["results_dir"]) / "training_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Resultats JSON : {results_json}")

    total_time = time.time() - total_start
    print(f"\n  Temps total : {total_time/60:.1f} minutes")
    print("\n" + "=" * 55)
    print("  Entraînement termine — prochaine etape : python src/evaluate.py")
    print("=" * 55)

    return model, history, test_metrics


# ─────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    config = CONFIG.copy()

    # Mode complet avec --full
    if "--full" in sys.argv:
        print("  Mode COMPLET : tout le dataset")
        config["max_samples"] = None
        config["phase1_epochs"] = 5
        config["phase2_epochs"] = 10
    else:
        print(f"  Mode RAPIDE : {config['max_samples']} images")
        print("  (lance avec --full pour le dataset complet)")

    train(config)