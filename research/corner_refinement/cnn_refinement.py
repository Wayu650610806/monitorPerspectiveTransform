"""
Method: CNN Regression Refinement
Approach: Uses deep learning backbones (EfficientNet-B0 or MobileNet-V2) to predict the exact corner coordinates within a crop.
Pros: Learns complex features that traditional OpenCV methods might miss.
"""

import cv2
import numpy as np
from pathlib import Path

# Global cache for models
models_cache = {}

def refine(crop, backbone='MobileNet'):
    """
    Refines corner position using CNN regression.
    Imports torch only when needed.
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    # Define model structure inside to ensure it's available for torch.load
    class CornerRefiner(nn.Module):
        def __init__(self, backbone_type='mobilenet'):
            super(CornerRefiner, self).__init__()
            if backbone_type == 'efficientnet':
                self.backbone = models.efficientnet_b0(pretrained=False)
                self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
            else:
                self.backbone = models.mobilenet_v2(pretrained=False)
                self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
        def forward(self, x): return self.backbone(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key = backbone.lower()
    
    if model_key not in models_cache:
        model_dir = Path(__file__).resolve().parent.parent / "models"
        if backbone == 'EfficientNet':
            path = model_dir / "EfficientB0_best.pth"
            m = CornerRefiner('efficientnet')
        else:
            path = model_dir / "MobileNetV2_best.pth"
            m = CornerRefiner('mobilenet')
            
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        m.load_state_dict(torch.load(str(path), map_location=device))
        m.to(device)
        m.eval()
        models_cache[model_key] = m

    model = models_cache[model_key]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    h, w = crop.shape[:2]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        coords = output.cpu().numpy()[0]
        
    # Scale back from normalized 0-1 (assuming model output is normalized)
    rx, ry = coords[0] * w, coords[1] * h
    return int(rx), int(ry)
