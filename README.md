# Monitor Perspective SDK

A professional tool for detecting monitor corners and applying perspective correction. Supports Manual, Auto (PicoDet + TinyPose), and YOLO Pose modes.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Note: If you have an NVIDIA GPU, install `paddlepaddle-gpu` instead of `paddlepaddle` for faster inference._

2.  **Add Models:**
    Place your model weights in the `models/` folder:
    - `models/best.pt` (YOLO Pose)
    - `models/picodet/` (PicoDet folder with `.pdmodel` and `.pdiparams`)
    - `models/tinypose/` (TinyPose folder with `.pdmodel` and `.pdiparams`)

## Usage

### Using the GUI

Run the application by providing a folder containing images:

```bash
python app.py "path/to/your/images"
```

**Controls:**

- `A` / `D`: Previous / Next image
- `M`: Cycle modes (Manual → Auto → YOLO)
- `R`: Rotate image 30°
- `C`: Reset rotation/points
- `SPACE`: Run detection (in Auto/YOLO modes)
- `Q`: Quit

### Integration (SDK)

If you are a developer and want to use this logic in your own Python project, follow these steps:

1. **Copy the Folders**: Copy `src/`, `models/`, and `config.py` into your project directory.
2. **Install Requirements**: Ensure you have installed the libraries listed in `requirements.txt`.
3. **Use the High-Level API**:
   The `get_warped_monitor` function is the main entry point. It defaults to the **YOLO mode**.

```python
import cv2
from src.detector import MonitorDetector

# 1. Initialize the detector (do this once at the start of your app)
detector = MonitorDetector()

# 2. Load your image
image = cv2.imread("monitor.jpg")

# 3. Get the warped result (one-call)
# This will find the corners and perform the perspective correction.
# By default, it uses YOLO mode for the best accuracy.
warped, corners = detector.get_warped_monitor(image)

if warped is not None:
    # 'warped' is an OpenCV BGR image (1280x800 by default)
    # You can now show it, save it, or process it further:
    cv2.imshow("Result", warped)
    cv2.waitKey(0)
    print(f"Success! Monitor corners found at: {corners}")
else:
    print("Monitor not detected in image.")
```

## Configuration

Modify `config.py` to change output resolution, detection thresholds, or default paths.

## 🔬 Research & Experiments (Experimental)

The `research/` folder is an isolated "experimental laboratory" where various alternative approaches were explored. These methods are provided for transparency and to showcase different technical possibilities; they are **not** part of the core production system.

### 1. Full Image OpenCV (`research/full_image_opencv/`)

This explores finding the monitor in a full-sized image using classical Computer Vision techniques without AI.

- **Methods included:**
  - `GrabCut`: Separates foreground object from background.
  - `Hough Segments`: Finds long vertical screen edges.
  - `Math Intersection`: Calculates precise corner points by intersecting infinite lines.
  - `Contour-Based`: Uses area and shape analysis (Adjusted version).
  - `UI Anchor`: Locks onto specific UI elements (tabs/buttons) as reference points.
- **To Run:** `python research/full_image_opencv/app.py "path/to/images"`

### 2. Corner Refinement Research (`research/corner_refinement/`)

This explores a high-precision **Two-Stage Approach**:

- **Stage 1:** Initial coarse detection using the stable YOLO Pose model.
- **Stage 2:** "Zooming in" on each detected corner to find the exact pixel location using 4 different refinement engines:
  - `OpenCV Features`: Shi-Tomasi/Harris corner detection.
  - `Smart Intersection`: Local line intersection within the crop.
  - `EfficientNet-B0`: Deep learning regression (CNN).
  - `MobileNet-V2`: Lightweight deep learning regression (CNN).
- **To Run:** `python research/corner_refinement/app.py "path/to/images"`

---

## 🛠️ Main System vs. Research

- **Main System (`app.py` / `src/`)**: Use this for production. It is stable, tested, and uses the best-performing AI model.
- **Research Folder (`research/`)**: Use this to see alternative ideas. It is completely isolated and does not affect the performance or stability of the main system.
