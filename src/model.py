"""
model.py
--------
Vision Transformer (ViT-B/16) adapté pour 6 canaux d'entrée (RGB + ELA).

Innovation principale :
  Le ViT standard attend 3 canaux. On modifie la première couche de projection
  pour accepter 6 canaux, en initialisant les 3 nouveaux canaux (ELA) avec les
  poids pré-entraînés des 3 canaux RGB — transfer learning partiel intelligent.

Architecture :
  Input (6, 224, 224)
    → Patch Embedding modifié (6→768)
    → 12 blocs Transformer (ViT-B/16)
    → [CLS] token
    → Classifier head (768 → 2)

Usage :
  python src/model.py   ← teste l'architecture et affiche un résumé
"""

import torch
import torch.nn as nn
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("  timm non disponible, utilisation de transformers")

try:
    from transformers import ViTForImageClassification, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ─────────────────────────────────────────
#  MODÈLE PRINCIPAL
# ─────────────────────────────────────────

class ForensicsViT(nn.Module):
    """
    ViT-B/16 modifié pour la détection de falsification d'images.

    Modifications par rapport au ViT standard :
      1. Patch embedding : 3 → 6 canaux d'entrée
      2. Tête de classification : 768 → 2 classes (authentic / tampered)
      3. Dropout supplémentaire avant le classifier (régularisation)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 6,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ── Chargement du ViT pré-entraîné (3 canaux) ──────────────
        if TIMM_AVAILABLE:
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0,       # on retire la tête originale
                in_chans=3           # on charge d'abord en 3 canaux
            )
            embed_dim = self.vit.embed_dim  # 768 pour ViT-B
        else:
            raise RuntimeError("timm requis : pip install timm")

        # ── Adaptation du Patch Embedding pour 6 canaux ─────────────
        # Récupère les poids originaux de la couche de projection (3 canaux)
        original_proj = self.vit.patch_embed.proj  # Conv2d(3, 768, 16, 16)
        original_weight = original_proj.weight.data  # (768, 3, 16, 16)

        # Crée une nouvelle Conv2d avec 6 canaux d'entrée
        new_proj = nn.Conv2d(
            in_channels=in_channels,  # 6
            out_channels=original_proj.out_channels,  # 768
            kernel_size=original_proj.kernel_size,    # 16
            stride=original_proj.stride,              # 16
            bias=original_proj.bias is not None
        )

        # Initialisation intelligente :
        # - Canaux 0-2 (RGB) : poids pré-entraînés originaux
        # - Canaux 3-5 (ELA) : copie des poids RGB × 0.5
        #   (ELA est une représentation dérivée de RGB, donc initialisation similaire)
        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = original_weight
            new_proj.weight[:, 3:, :, :] = original_weight * 0.5
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data.clone()

        # Remplacement dans le modèle
        self.vit.patch_embed.proj = new_proj

        # ── Tête de classification forensique ───────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

        print(f"  ForensicsViT initialisé :")
        print(f"    Canaux d'entrée  : {in_channels} (RGB + ELA)")
        print(f"    Embed dim        : {embed_dim}")
        print(f"    Classes          : {num_classes} (authentic / tampered)")
        print(f"    Pré-entraîné     : {pretrained}")

    def forward(self, x):
        """
        Args:
            x : tensor (batch, 6, 224, 224)
        Returns:
            logits : tensor (batch, 2)
        """
        # Extraction des features via ViT (retourne le CLS token)
        features = self.vit(x)   # (batch, 768)

        # Classification forensique
        logits = self.classifier(features)  # (batch, 2)

        return logits

    def get_attention_maps(self, x):
        """
        Extrait les cartes d'attention du dernier bloc Transformer.
        Utilisé pour la visualisation Grad-CAM / attention rollout.

        Returns:
            attention : tensor (batch, heads, seq_len, seq_len)
        """
        attention_maps = []

        def hook_fn(module, input, output):
            attention_maps.append(output)

        # Hook sur le dernier bloc d'attention
        last_block = self.vit.blocks[-1].attn
        hook = last_block.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.forward(x)

        hook.remove()

        if attention_maps:
            return attention_maps[0]
        return None

    def freeze_backbone(self, freeze: bool = True):
        """
        Gèle/dégèle les poids du backbone ViT.
        Utile pour l'entraînement en deux phases :
          Phase 1 : backbone gelé, on entraîne seulement le classifier
          Phase 2 : fine-tuning complet avec lr très faible
        """
        for param in self.vit.parameters():
            param.requires_grad = not freeze

        # Le patch embedding modifié doit toujours être entraînable
        for param in self.vit.patch_embed.proj.parameters():
            param.requires_grad = True

        status = "gelé" if freeze else "dégelé"
        print(f"  Backbone ViT {status} (patch embedding toujours entraînable)")

    def count_parameters(self):
        """Retourne le nombre de paramètres entraînables / total."""
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


# ─────────────────────────────────────────
#  FONCTION DE CRÉATION RAPIDE
# ─────────────────────────────────────────

def create_model(pretrained: bool = True, dropout: float = 0.3) -> ForensicsViT:
    """Factory function — crée et retourne le modèle prêt à l'emploi."""
    model = ForensicsViT(
        num_classes=2,
        in_channels=6,
        dropout=dropout,
        pretrained=pretrained
    )
    return model


# ─────────────────────────────────────────
#  TEST DE L'ARCHITECTURE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Test architecture ForensicsViT")
    print("=" * 55)

    print("\nChargement du modèle (téléchargement poids si 1ère fois)...")
    model = create_model(pretrained=True)

    # ── Résumé des paramètres ────────────────────────────────────
    trainable, total = model.count_parameters()
    print(f"\n  Paramètres entraînables : {trainable:,}")
    print(f"  Paramètres totaux       : {total:,}")
    print(f"  Ratio entraînable       : {trainable/total*100:.1f}%")

    # ── Test d'un forward pass ───────────────────────────────────
    print("\nTest forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 6, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape   : {dummy_input.shape}")
    print(f"  Output shape  : {output.shape}")
    print(f"  Logits sample : {output[0].tolist()}")

    # Probabilités via softmax
    probs = torch.softmax(output, dim=1)
    print(f"  Proba sample  : authentic={probs[0][0]:.3f}, tampered={probs[0][1]:.3f}")

    # ── Test freeze/unfreeze ─────────────────────────────────────
    print("\nTest freeze backbone...")
    model.freeze_backbone(freeze=True)
    trainable_frozen, _ = model.count_parameters()
    print(f"  Params entraînables (backbone gelé)  : {trainable_frozen:,}")

    model.freeze_backbone(freeze=False)
    trainable_full, _ = model.count_parameters()
    print(f"  Params entraînables (fine-tuning)    : {trainable_full:,}")

    # ── Vérification du patch embedding modifié ──────────────────
    print("\nVérification du patch embedding 6 canaux...")
    proj = model.vit.patch_embed.proj
    print(f"  Conv2d : in={proj.in_channels}, out={proj.out_channels}, "
          f"kernel={proj.kernel_size}")
    assert proj.in_channels == 6, "ERREUR : patch embedding pas modifié !"
    print(f"  Poids RGB  (canal 0) norm : {proj.weight[:, 0, :, :].norm():.4f}")
    print(f"  Poids ELA  (canal 3) norm : {proj.weight[:, 3, :, :].norm():.4f}")
    print(f"  Ratio ELA/RGB             : {proj.weight[:, 3, :, :].norm() / proj.weight[:, 0, :, :].norm():.4f} (attendu ~0.5)")

    print("\n" + "=" * 55)
    print("  Modèle OK — prochaine étape : python src/train.py")
    print("=" * 55)