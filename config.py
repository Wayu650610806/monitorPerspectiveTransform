import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Model Paths (Relative to the project root)
MODELS_DIR      = BASE_DIR / "models"
PICODET_DIR     = MODELS_DIR / "picodet"
TINYPOSE_DIR    = MODELS_DIR / "tinypose"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"

# Output Settings
OUTPUT_W        = 1280
OUTPUT_H        = 800
USE_GPU         = False                 # Set to True if GPU + paddlepaddle-gpu/torch-gpu are available

# Detection Settings
BBOX_EXPAND     = 1.5                   # Expand bbox before TinyPose
BBOX_THRESHOLD  = 0.4                   # Minimum confidence for PicoDet

# GUI Settings
ROTATE_R_STEP   = 30
ROTATE_E_STEP   = 45
POINT_RADIUS    = 8
POINT_COLOR     = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)] # TL, TR, BR, BL
LABEL_TEXT      = ["TL", "TR", "BR", "BL"]
LINE_COLOR      = (0, 255, 255)
BBOX_COLOR      = (255, 200, 0)

# Supported Extensions
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
