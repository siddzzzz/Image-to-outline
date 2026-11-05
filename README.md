# Image-to-Outline Vector Engine

Ultra-lightweight Python engine that isolates foreground entities (people, objects, banners, text) from images and converts them into smooth, animation-style black-and-white **Cubic Bézier SVG Path equations** (`{"w": 512, "h": 512, "d": "M..."}`) for compact database storage.

## Key Features
- **Lightweight Foreground Isolation (~4.5 MB):** Uses `u2netp.onnx` via CPU `onnxruntime` (no PyTorch or GPU needed) with guided-filter edge snapping and multi-region tiling.
- **Natural Comic/Animation Face Outlines (~227 KB):** Automatically detects human faces via YuNet ONNX to extract clean eyebrows, eyelid arcs, pupil dots, lip parting contours, and jawlines while silencing cheek & forehead shadow noise.
- **Smooth Animated Line Art:** Combines a bold outer silhouette contour with multi-pass bilateral surface smoothing, Canny structural strokes, and Difference-of-Gaussians (DoG) ink detection.
- **Tiny Database Footprint (2 KB – 15 KB):** Traces contours into piecewise Cubic Bézier curves (`M`, `L`, `C`, `Z`), cutting storage size by **85%–98%** compared to storing raster images.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(The 4.5 MB `u2netp.onnx` segmentation model and 227 KB `face_detection_yunet_2023mar.onnx` model are downloaded automatically into `models/` on first run.)*

### 2. Run Interactive Web Studio & FastAPI Backend
```bash
python api.py
# or: uvicorn api:app --reload --port 8085
```
Open **http://localhost:8085** in your browser to upload images, tweak detail presets (`minimal`, `medium`, `high`), and inspect the database JSON payload.

### 3. Command-Line Interface (CLI)
```bash
python cli.py path/to/image.jpg --detail medium
```
Outputs both `image.outline.json` (compact DB payload) and `image.outline.svg` (standalone vector file).

### 4. Python SDK Usage
```python
from outline_engine import ImageToOutlinePipeline

pipeline = ImageToOutlinePipeline(max_dimension=512)
result = pipeline.process("photo.jpg", detail="medium")

# Save compact JSON to your database (typically 3 KB - 10 KB)
db_json_string = result.to_db_json()
# -> '{"w":512,"h":512,"d":"M214 42C225 42..."}'
```

### 5. Rendering the Database Payload on Frontend
```html
<svg viewBox="0 0 512 512" width="100%" height="100%">
  <path d="M214 42C225 42..." fill="#000000" fill-rule="evenodd" />
</svg>
```
