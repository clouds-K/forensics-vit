import torch
import torchvision
try:
    import transformers
except ImportError:
    transformers = None
import timm
import cv2
import matplotlib
from PIL import Image

print("=" * 40)
print(f"PyTorch       : {torch.__version__}")
print(f"Torchvision   : {torchvision.__version__}")
print(f"Transformers  : {transformers.__version__ if transformers else 'not installed (optional)'}")
print(f"timm          : {timm.__version__}")
print(f"OpenCV        : {cv2.__version__}")
print(f"Matplotlib    : {matplotlib.__version__}")
print(f"Pillow        : {Image.__version__}")
print("=" * 40)
print("Environment OK")