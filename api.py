from pathlib import Path
from typing import Literal
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from outline_engine import ImageToOutlinePipeline

app = FastAPI(
    title="Image-to-Outline Vector Engine API",
    description="Converts foreground entities in images into ultra-compact Cubic Bézier SVG path JSON.",
    version="1.0.0",
)

pipeline = ImageToOutlinePipeline(max_dimension=512)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
def preload_model():
    # Ensure both lightweight models are downloaded and warmed up on startup
    try:
        pipeline.segmenter.ensure_model()
        pipeline.stylizer.face_enhancer.ensure_model()
    except Exception as exc:
        print(f"[Warning] Could not pre-download model on startup: {exc}")


@app.get("/", response_class=HTMLResponse)
def serve_studio():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Studio UI not found.</h3>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.post("/api/outline")
async def convert_image_to_outline(
    file: UploadFile = File(...),
    detail: Literal["minimal", "medium", "high"] = Form("medium"),
    vector_mode: Literal["centerline", "tapered_ribbon", "ribbon"] = Form("centerline"),
    max_dimension: int = Form(680),
    simplify_epsilon: float = Form(0.65),
    outer_thickness: int = Form(2),
    remove_background: bool = Form(True),
    auto_crop: bool = Form(True),
    enhance_faces: bool = Form(True),
):
    """
    Main endpoint for converting an uploaded image into a compact DB vector outline.
    Returns both the compact DB payload (`db_payload`: {"w", "h", "d", ["sw"]}) and metrics.
    """
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    try:
        res = pipeline.process(
            image_input=raw_bytes,
            detail=detail,
            max_dimension=max(128, min(1200, max_dimension)),
            simplify_epsilon=max(0.3, min(3.0, simplify_epsilon)),
            outer_thickness=max(1, min(6, outer_thickness)),
            remove_background=remove_background,
            auto_crop=auto_crop,
            enhance_faces=enhance_faces,
            vector_mode=vector_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {exc}") from exc

    orig_kb = len(raw_bytes) / 1024.0
    json_kb = res.json_size_bytes / 1024.0
    savings_pct = round((1.0 - (json_kb / max(orig_kb, 0.01))) * 100.0, 1)

    return JSONResponse({
        "db_payload": res.to_db_dict(),
        "metrics": {
            "processing_ms": res.processing_ms,
            "original_size_kb": round(orig_kb, 2),
            "json_size_kb": round(json_kb, 2),
            "gzip_size_kb": round(res.gzip_size_bytes / 1024.0, 2),
            "savings_percent": savings_pct,
            "contours": res.contours_count,
            "anchor_points": res.anchor_points_count,
        },
        "svg": res.svg_string,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8085, reload=True)


