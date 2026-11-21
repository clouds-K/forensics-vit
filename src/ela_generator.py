"""
ela_generator.py
----------------
Génère les cartes ELA (Error Level Analysis) pour le projet forensics.

Principe :
  1. On re-compresse l'image en JPEG à qualité connue (95%)
  2. On calcule la différence pixel par pixel avec l'originale
  3. On amplifie le résultat pour le rendre visible
  → Les zones manipulées apparaissent plus lumineuses (historique de compression différent)

Usage :
  python src/ela_generator.py
"""

import os
import sys
import hashlib
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────
#  FONCTION PRINCIPALE : générer une ELA
# ─────────────────────────────────────────

def generate_ela(image_path: str, quality: int = 95) -> tuple[Image.Image, float]:
    """
    Génère la carte ELA d'une image.

    Args:
        image_path : chemin vers l'image source
        quality    : qualité JPEG de re-compression (95 par défaut)

    Returns:
        ela_image  : image PIL de la carte ELA amplifiée
        max_diff   : valeur max de différence (indicateur de manipulation)
    """
    original = Image.open(image_path).convert('RGB')

    # Re-compression temporaire
    temp_path = str(image_path) + "_temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    recompressed = Image.open(temp_path)

    # Différence pixel par pixel
    ela_raw = ImageChops.difference(original, recompressed)

    # Calcul du max de différence (métrique forensique)
    ela_array = np.array(ela_raw)
    max_diff = float(ela_array.max())

    # Amplification pour visualisation
    scale = 255.0 / max_diff if max_diff > 0 else 1.0
    ela_image = ImageEnhance.Brightness(ela_raw).enhance(scale)

    # Nettoyage fichier temporaire
    os.remove(temp_path)

    return ela_image, max_diff


# ─────────────────────────────────────────
#  HASH SHA-256 (chaîne de custody)
# ─────────────────────────────────────────

def compute_sha256(image_path: str) -> str:
    """
    Calcule le hash SHA-256 d'un fichier image.
    Utilisé pour garantir l'intégrité de la preuve numérique.
    """
    sha256 = hashlib.sha256()
    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─────────────────────────────────────────
#  VISUALISATION : original + ELA côte à côte
# ─────────────────────────────────────────

def visualize_ela(image_path: str, output_path: str = None):
    """
    Affiche et sauvegarde la comparaison Original / ELA.
    C'est ce que tu montreras en soutenance.
    """
    original = Image.open(image_path).convert('RGB')
    ela_image, max_diff = generate_ela(image_path)
    sha = compute_sha256(image_path)

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(
        f"Analyse ELA — {Path(image_path).name}\n"
        f"SHA-256 : {sha[:32]}...  |  Max diff : {max_diff:.1f}/255",
        fontsize=10, color='#333333'
    )

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

    # Image originale
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original)
    ax1.set_title("Image originale", fontsize=11)
    ax1.axis('off')

    # Carte ELA
    ax2 = fig.add_subplot(gs[1])
    ela_np = np.array(ela_image)
    im = ax2.imshow(ela_np, cmap='hot')
    ax2.set_title(
        "Carte ELA\n(zones claires = potentiellement manipulées)",
        fontsize=11
    )
    ax2.axis('off')

    # Colorbar
    ax3 = fig.add_subplot(gs[2])
    plt.colorbar(im, cax=ax3, label='Intensité de différence')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"  Visualisation sauvegardée : {output_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  TRAITEMENT EN BATCH : tout un dossier
# ─────────────────────────────────────────

def process_folder(input_dir: str, output_dir: str, label: str = ""):
    """
    Génère les ELA pour toutes les images d'un dossier.

    Args:
        input_dir  : dossier source (ex: data/raw/authentic)
        output_dir : dossier destination (ex: data/ela/authentic)
        label      : "authentic" ou "tampered" (pour les logs)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"  Aucune image trouvée dans {input_dir}")
        return []

    print(f"\n  Traitement de {len(images)} images [{label}]")
    results = []

    for i, img_path in enumerate(images):
        try:
            ela_img, max_diff = generate_ela(str(img_path))
            sha = compute_sha256(str(img_path))

            # Sauvegarde ELA
            out_filename = img_path.stem + '_ela.png'
            out_file = output_path / out_filename
            ela_img.save(str(out_file))

            results.append({
                'filename'  : img_path.name,
                'ela_path'  : str(out_file),
                'max_diff'  : max_diff,
                'sha256'    : sha,
                'label'     : label
            })

            # Progression
            if (i + 1) % 50 == 0 or (i + 1) == len(images):
                print(f"  [{i+1:4d}/{len(images)}] OK — dernier : {img_path.name} (max_diff={max_diff:.1f})")

        except Exception as e:
            print(f"  ERREUR sur {img_path.name} : {e}")

    return results


# ─────────────────────────────────────────
#  SCRIPT PRINCIPAL
# ─────────────────────────────────────────

if __name__ == "__main__":

    BASE = Path(__file__).parent.parent  # racine du projet

    print("=" * 55)
    print("  ELA Generator — Forensics Image Analysis")
    print("=" * 55)

    # ── Mode TEST : une seule image ──────────────────────
    # Lance avec : python src/ela_generator.py test <chemin_image>
    if len(sys.argv) == 3 and sys.argv[1] == "test":
        img = sys.argv[2]
        print(f"\nTest ELA sur : {img}")
        sha = compute_sha256(img)
        print(f"  SHA-256 : {sha}")
        visualize_ela(img, output_path=str(BASE / "reports" / "test_ela.png"))
        sys.exit(0)

    # ── Mode BATCH : tout le dataset ─────────────────────
    authentic_dir = BASE / "data" / "raw" / "authentic"
    tampered_dir  = BASE / "data" / "raw" / "tampered"
    ela_auth_dir  = BASE / "data" / "ela" / "authentic"
    ela_tamp_dir  = BASE / "data" / "ela" / "tampered"

    # Vérification que le dataset est présent
    if not any(authentic_dir.iterdir() if authentic_dir.exists() else []):
        print("\n  ATTENTION : data/raw/authentic/ est vide.")
        print("  Télécharge le dataset CASIA v2 depuis :")
        print("  https://www.kaggle.com/datasets/sophatvathana/casia-dataset")
        print("\n  Pour tester avec une seule image :")
        print("  python src/ela_generator.py test <chemin_vers_image.jpg>")
        sys.exit(1)

    # Traitement batch
    all_results = []
    all_results += process_folder(str(authentic_dir), str(ela_auth_dir), label="authentic")
    all_results += process_folder(str(tampered_dir),  str(ela_tamp_dir), label="tampered")

    # Résumé
    print("\n" + "=" * 55)
    print(f"  Total traité   : {len(all_results)} images")
    auth_count = sum(1 for r in all_results if r['label'] == 'authentic')
    tamp_count = sum(1 for r in all_results if r['label'] == 'tampered')
    print(f"  Authentiques   : {auth_count}")
    print(f"  Falsifiées     : {tamp_count}")

    if all_results:
        diffs = [r['max_diff'] for r in all_results]
        print(f"  Max diff moyen : {sum(diffs)/len(diffs):.1f}")
        print(f"  Max diff max   : {max(diffs):.1f}")

    print("=" * 55)
    print("  ELA générées dans data/ela/")
  