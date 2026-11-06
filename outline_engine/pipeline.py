from dataclasses import dataclass
from pathlib import Path
import time
import json
import cv2
import numpy as np

from .segmenter import ForegroundSegmenter
from .stylizer import AnimatedLineStylizer, DetailLevel
from .vectorizer import BezierVectorizer, VectorMode


@dataclass
class OutlineResult:
    """Result object returned by ImageToOutlinePipeline."""
    width: int
    height: int
    detail: DetailLevel
    path_d: str
    contours_count: int
    anchor_points_count: int
    processing_ms: float
    json_size_bytes: int
    gzip_size_bytes: int
    svg_string: str
    vector_mode: VectorMode = "centerline"
    stroke_width: float | None = None

    def to_db_dict(self) -> dict:
        """Minimal JSON object designed to be stored directly in a database column."""
        payload = {
            "w": self.width,
            "h": self.height,
            "d": self.path_d,
        }
        if self.vector_mode == "centerline" and self.stroke_width is not None:
            payload["sw"] = self.stroke_width
        return payload

    def to_db_json(self) -> str:
        """Compact serialized JSON string with no extra whitespace."""
        return json.dumps(self.to_db_dict(), separators=(",", ":"))


class ImageToOutlinePipeline:
    """
    End-to-end lightweight pipeline that isolates foreground entities and converts
    them into smooth, animation-style black-and-white Cubic Bézier vector outlines.
    """

    def __init__(self, max_dimension: int = 680):
        self.max_dimension = max_dimension
        self.segmenter = ForegroundSegmenter()
        self.stylizer = AnimatedLineStylizer()

    def _decode_image(
        self, image_input: str | Path | bytes | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(image_input, (str, Path)):
            raw = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError("Could not decode image bytes.")
        elif isinstance(image_input, np.ndarray):
            raw = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        alpha = None
        if raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
            bgr = (raw[:, :, :3].astype(np.float32) * alpha_f + 255.0 * (1.0 - alpha_f)).astype(np.uint8)
        else:
            bgr = raw[:, :, :3]

        # Pre-cap very large camera photos to 1280px before initial segmentation
        h, w = bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / float(max(h, w))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return bgr, alpha

    @staticmethod
    def _crop_to_foreground(
        bgr: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crops tightly around the foreground subject(s) with a clean padding margin
        so distant subjects or group photos fill the canvas and retain maximum facial detail.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return bgr, mask

        h, w = bgr.shape[:2]
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        y1, y2 = int(np.min(ys)), int(np.max(ys))

        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        pad_x = max(12, int(round(box_w * pad_ratio)))
        pad_y = max(12, int(round(box_h * pad_ratio)))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def process(
        self,
        image_input: str | Path | bytes | np.ndarray,
        detail: DetailLevel = "medium",
        max_dimension: int | None = None,
        simplify_epsilon: float = 0.65,
        outer_thickness: int | None = None,
        remove_background: bool = True,
        auto_crop: bool = True,
        enhance_faces: bool = True,
        vector_mode: VectorMode = "centerline",
    ) -> OutlineResult:
        """
        Runs the full pipeline on an image and returns an OutlineResult.
        """
        t0 = time.perf_counter()
        target_dim = max_dimension or self.max_dimension

        # Stage 1: Decode image
        bgr, alpha = self._decode_image(image_input)
        h, w = bgr.shape[:2]

        # Stage 2: Foreground segmentation (U2Netp ONNX + uniform BG text/banner hybrid)
        if remove_background:
            fg_mask = self.segmenter.extract_mask(bgr, alpha_channel=alpha)
            if auto_crop:
                bgr, fg_mask = self._crop_to_foreground(bgr, fg_mask, pad_ratio=0.05)
        else:
            fg_mask = np.full((h, w), 255, dtype=np.uint8)

        # Normalize subject canvas to target_dim (so small/distant subjects get full resolution)
        ch, cw = bgr.shape[:2]
        scale = float(target_dim) / float(max(ch, cw))
        if abs(scale - 1.0) > 0.05:
            new_w = max(32, int(round(cw * scale)))
            new_h = max(32, int(round(ch * scale)))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
            fg_mask = cv2.resize(fg_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Stage 3: Smooth animation-style black & white line extraction (with face enhancement)
        line_bitmap = self.stylizer.generate_line_bitmap(
            bgr_img=bgr,
            fg_mask=fg_mask,
            detail=detail,
            outer_line_thickness=outer_thickness,
            enhance_faces=enhance_faces,
        )

        # Stage 4: Mathematical Cubic Bézier vectorization
        vectorizer = BezierVectorizer(
            simplify_epsilon=simplify_epsilon,
            precision=0,
            mode=vector_mode,
        )
        traced = vectorizer.trace_bitmap(line_bitmap, mode=vector_mode)
        svg_str = vectorizer.to_svg_string(traced)

        db_dict = {"w": traced["w"], "h": traced["h"], "d": traced["d"]}
        if traced.get("mode") == "centerline" and traced.get("stroke_width") is not None:
            db_dict["sw"] = traced["stroke_width"]

        db_json_str = json.dumps(db_dict, separators=(",", ":"))
        gzip_b64 = vectorizer.compress_payload(db_dict)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return OutlineResult(
            width=traced["w"],
            height=traced["h"],
            detail=detail,
            path_d=traced["d"],
            contours_count=traced["contours"],
            anchor_points_count=traced["points"],
            processing_ms=round(elapsed_ms, 1),
            json_size_bytes=len(db_json_str.encode("utf-8")),
            gzip_size_bytes=len(gzip_b64.encode("ascii")) * 3 // 4,
            svg_string=svg_str,
            vector_mode=traced.get("mode", vector_mode),
            stroke_width=traced.get("stroke_width"),
        )
