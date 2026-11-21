"""
dataset.py
----------
Dataset PyTorch pour le projet forensics.

Charge chaque échantillon sous forme de :
  - image RGB originale  (3 canaux)
  - carte ELA            (3 canaux)
  → concaténés en tensor 6 canaux → entrée du ViT

Label : 0 = authentic, 1 = tampered

Usage :
  python src/dataset.py   ← lance un test de santé du dataset
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ─────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────

# ViT-B/16 attend des images 224×224
IMAGE_SIZE = 224

# Normalisation ImageNet (on part d'un modèle pré-entraîné)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(mode: str = "train"):
    """
    Transforms pour les images RGB (3 canaux).
    mode : 'train' (avec augmentation) ou 'val'/'test'
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def get_ela_transforms(mode: str = "train"):
    """
    Transforms pour les cartes ELA (3 canaux, pas de ColorJitter).
    On garde les mêmes flip que l'image RGB pour la cohérence spatiale.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            # Normalisation ELA : centrée sur 0.5
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


# ─────────────────────────────────────────
#  DATASET PRINCIPAL
# ─────────────────────────────────────────

class ForensicsDataset(Dataset):
    """
    Dataset forensics CASIA v2.

    Chaque item retourne :
      - tensor (6, 224, 224) : RGB + ELA concaténés
      - label int            : 0 = authentic, 1 = tampered
      - filename str         : nom du fichier (pour le rapport forensique)
    """

    def __init__(
        self,
        raw_dir: str,
        ela_dir: str,
        mode: str = "train",
        max_samples: int = None
    ):
        """
        Args:
            raw_dir    : dossier racine des images originales (contient authentic/ et tampered/)
            ela_dir    : dossier racine des cartes ELA        (contient authentic/ et tampered/)
            mode       : 'train', 'val', ou 'test'
            max_samples: limite le nombre d'images (utile pour tests rapides)
        """
        self.raw_dir = Path(raw_dir)
        self.ela_dir = Path(ela_dir)
        self.mode    = mode

        self.img_transform = get_transforms(mode)
        self.ela_transform = get_ela_transforms(mode)

        # Construction de l'index : liste de (raw_path, ela_path, label)
        self.samples = []
        self._build_index(max_samples)

        print(f"  Dataset [{mode}] : {len(self.samples)} échantillons "
              f"({sum(1 for _,_,l in self.samples if l==0)} authentic, "
              f"{sum(1 for _,_,l in self.samples if l==1)} tampered)")

    def _build_index(self, max_samples):
        """Construit la liste des paires (image, ela, label)."""
        extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

        for label, category in [(0, 'authentic'), (1, 'tampered')]:
            raw_folder = self.raw_dir / category
            ela_folder = self.ela_dir / category

            if not raw_folder.exists():
                print(f"  ATTENTION : {raw_folder} introuvable, ignoré.")
                continue

            images = sorted([
                f for f in raw_folder.iterdir()
                if f.suffix.lower() in extensions
            ])

            for img_path in images:
                # Cherche la carte ELA correspondante
                ela_path = ela_folder / (img_path.stem + '_ela.png')

                if not ela_path.exists():
                    continue  # ELA manquante, on saute

                self.samples.append((img_path, ela_path, label))

        # Limite optionnelle pour tests rapides
        if max_samples and len(self.samples) > max_samples:
            # On garde la proportion authentic/tampered
            import random
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, ela_path, label = self.samples[idx]

        # Chargement images
        try:
            img_rgb = Image.open(raw_path).convert('RGB')
            img_ela = Image.open(ela_path).convert('RGB')
        except Exception as e:
            print(f"  Erreur chargement {raw_path.name} : {e}")
            # Retourne un item vide plutôt que de crasher
            return torch.zeros(6, IMAGE_SIZE, IMAGE_SIZE), 0, "error"

        # Application des transforms
        # Note : pour la cohérence spatiale train (mêmes flips sur RGB et ELA),
        # on utilise un seed fixe par item
        if self.mode == "train":
            seed = idx
            torch.manual_seed(seed)
            tensor_rgb = self.img_transform(img_rgb)
            torch.manual_seed(seed)
            tensor_ela = self.ela_transform(img_ela)
        else:
            tensor_rgb = self.img_transform(img_rgb)
            tensor_ela = self.ela_transform(img_ela)

        # Concaténation : (3, H, W) + (3, H, W) → (6, H, W)
        tensor_combined = torch.cat([tensor_rgb, tensor_ela], dim=0)

        return tensor_combined, label, raw_path.name


# ─────────────────────────────────────────
#  FACTORY : train / val / test splits
# ─────────────────────────────────────────

def create_dataloaders(
    raw_dir: str,
    ela_dir: str,
    batch_size: int = 16,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    num_workers: int = 0,
    max_samples: int = None
):
    """
    Crée les DataLoaders train / val / test.

    Splits : 75% train · 15% val · 10% test (par défaut)
    num_workers=0 sur Windows (évite les erreurs de multiprocessing)

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # Dataset complet en mode train (on splittera après)
    full_dataset = ForensicsDataset(raw_dir, ela_dir, mode="train", max_samples=max_samples)

    total = len(full_dataset)
    n_test = int(total * test_ratio)
    n_val  = int(total * val_ratio)
    n_train = total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Forcer le mode val/test sur les sous-datasets
    # (désactive les augmentations data)
    val_ds.dataset.mode  = "val"
    test_ds.dataset.mode = "test"

    print(f"\n  Splits : train={n_train} · val={n_val} · test={n_test}")

    # Calcul des poids de classe (dataset déséquilibré : 7491 vs 1827)
    labels = [full_dataset.samples[i][2] for i in range(total)]
    n_authentic = labels.count(0)
    n_tampered  = labels.count(1)
    weight_authentic = total / (2 * n_authentic)
    weight_tampered  = total / (2 * n_tampered)
    class_weights = torch.tensor([weight_authentic, weight_tampered], dtype=torch.float)
    print(f"  Poids de classe : authentic={weight_authentic:.3f} · tampered={weight_tampered:.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_weights


# ─────────────────────────────────────────
#  TEST DE SANTÉ DU DATASET
# ─────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BASE = Path(__file__).parent.parent

    raw_dir = BASE / "data" / "raw"
    ela_dir = BASE / "data" / "ela"

    print("=" * 55)
    print("  Test de santé du Dataset")
    print("=" * 55)

    # Test rapide sur 200 images
    print("\nChargement d'un sous-ensemble de 200 images...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        str(raw_dir), str(ela_dir),
        batch_size=4,
        max_samples=200
    )

    # Récupère un batch
    batch_tensors, batch_labels, batch_names = next(iter(train_loader))

    print(f"\n  Shape d'un batch  : {batch_tensors.shape}")
    print(f"  Canaux            : {batch_tensors.shape[1]} (3 RGB + 3 ELA)")
    print(f"  Labels            : {batch_labels.tolist()}")
    print(f"  Fichiers          : {list(batch_names)}")
    print(f"  Dtype             : {batch_tensors.dtype}")
    print(f"  Poids de classe   : {class_weights.tolist()}")

    # Visualisation d'un échantillon
    print("\nGénération de la visualisation...")

    sample_tensor = batch_tensors[0]     # (6, 224, 224)
    sample_label  = batch_labels[0].item()
    sample_name   = batch_names[0]

    # Dénormalisation pour affichage
    mean_rgb = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_rgb  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    rgb_display = sample_tensor[:3] * std_rgb + mean_rgb
    rgb_display = rgb_display.permute(1, 2, 0).numpy().clip(0, 1)

    ela_display = sample_tensor[3:] * 0.5 + 0.5
    ela_display = ela_display.permute(1, 2, 0).numpy().clip(0, 1)

    label_str = "TAMPERED 🔴" if sample_label == 1 else "AUTHENTIC 🟢"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{sample_name}\nLabel : {label_str}", fontsize=12)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Image RGB originale")
    axes[0].axis('off')

    axes[1].imshow(ela_display, cmap='hot')
    axes[1].set_title("Carte ELA (zones claires = suspect)")
    axes[1].axis('off')

    plt.tight_layout()
    out = BASE / "reports" / "dataset_sample.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"\n  Visualisation sauvegardée : {out}")
    plt.show()
    plt.close()

    print("\n" + "=" * 55)
    print("  Dataset OK — prochaine étape : python src/model.py")
    print("=" * 55)