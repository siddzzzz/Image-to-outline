from typing import Literal
import cv2
import numpy as np

from .face_enhancer import FaceEnhancer

DetailLevel = Literal["minimal", "medium", "high"]


class AnimatedLineStylizer:
    """
    Converts a masked BGR image into a smooth, animation-style binary line-art bitmap
    (255 = black ink line, 0 = white background).

    Features:
    1. Dedicated Face Enhancement: Extracts natural, comic-style eyebrows, eyelids, pupils,
       lips, and jawlines while silencing cheek & forehead shadow noise.
    2. Multi-pass Bilateral Cartoon Filtering: Flattens surface shading & lighting gradients.
    3. XDoG Valley Detection: Extracts dark creases, folds, and seams as clean single strokes.
    4. Zero-Gap Silhouette Fusion: Internal lines clip directly to the guided-filtered
       mask boundary and merge seamlessly with the outer silhouette.
    """

    def __init__(self):
        self.face_enhancer = FaceEnhancer()

    PRESETS = {
        "minimal": {
            "bilateral_passes": 3,
            "bilateral_sigma_color": 48,
            "bilateral_sigma_space": 26,
            "outer_thickness": 2,
            "valley_sigma_bg": 2.8,
            "valley_sigma_fg": 0.8,
            "valley_thresh": 7.0,
            "canny_low": 70,
            "canny_high": 160,
            "min_component_area": 30,
        },
        "medium": {
            "bilateral_passes": 3,
            "bilateral_sigma_color": 38,
            "bilateral_sigma_space": 22,
            "outer_thickness": 2,
            "valley_sigma_bg": 2.8,
            "valley_sigma_fg": 0.7,
            "valley_thresh": 5.0,
            "canny_low": 50,
            "canny_high": 125,
            "min_component_area": 18,
        },
        "high": {
            "bilateral_passes": 2,
            "bilateral_sigma_color": 28,
            "bilateral_sigma_space": 14,
            "outer_thickness": 2,
            "valley_sigma_bg": 2.4,
            "valley_sigma_fg": 0.6,
            "valley_thresh": 3.6,
            "canny_low": 35,
            "canny_high": 95,
            "min_component_area": 10,
        },
    }

    def generate_line_bitmap(
        self,
        bgr_img: np.ndarray,
        fg_mask: np.ndarray,
        detail: DetailLevel = "medium",
        outer_line_thickness: int | None = None,
        enhance_faces: bool = True,
    ) -> np.ndarray:
        """
        Produces a clean binary uint8 mask where 255 represents the black outline strokes
        and 0 represents white empty space.
        """
        cfg = self.PRESETS.get(detail, self.PRESETS["medium"])
        outer_thick = outer_line_thickness if outer_line_thickness is not None else cfg["outer_thickness"]

        # 1. Smooth foreground mask slightly for fluid outer curves
        smooth_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)

        # 2. Extract Bold Outer Silhouette right on the edge-snapped mask boundary
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_thick * 2 + 1, outer_thick * 2 + 1))
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        outer_silhouette = cv2.subtract(
            cv2.dilate(smooth_mask, k_out, iterations=1),
            cv2.erode(smooth_mask, k_in, iterations=1),
        )

        # 3. Detect and extract natural facial features & skin suppression mask
        face_lines = None
        skin_mask = None
        if enhance_faces:
            face_lines, skin_mask = self.face_enhancer.extract_face_features(bgr_img, detail=detail)

        # 4. Cartoon Surface Smoothing (flattens surface gradients while keeping main features sharp)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        smoothed = gray
        for _ in range(cfg["bilateral_passes"]):
            smoothed = cv2.bilateralFilter(
                smoothed,
                d=7,
                sigmaColor=cfg["bilateral_sigma_color"],
                sigmaSpace=cfg["bilateral_sigma_space"],
            )

        # 5a. XDoG Local Valley Detector: extracts dark features, creases, and seams
        local_bg = cv2.GaussianBlur(smoothed, (0, 0), cfg["valley_sigma_bg"])
        local_fg = cv2.GaussianBlur(smoothed, (0, 0), cfg["valley_sigma_fg"])
        diff = local_bg.astype(np.float32) - local_fg.astype(np.float32)
        ink_valleys = (diff > cfg["valley_thresh"]).astype(np.uint8) * 255

        # 5b. Structural Canny edges for clean boundaries between major color blocks
        strong_edges = cv2.Canny(smoothed, cfg["canny_low"], cfg["canny_high"], L2gradient=True)

        # 6. Combine & Clip Directly to `smooth_mask` (Zero Gap to Outer Silhouette!)
        combined_inner = cv2.bitwise_or(ink_valleys, strong_edges)
        combined_inner = cv2.bitwise_and(combined_inner, smooth_mask)

        # Suppress cheek and forehead shadow noise if faces are present
        if skin_mask is not None and cv2.countNonZero(skin_mask) > 0:
            combined_inner[skin_mask > 0] = 0

        # Bridge micro-breaks in lines before speckle removal
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined_inner = cv2.dilate(combined_inner, k_bridge, iterations=1)

        # 7. Prune tiny isolated speckles while preserving long thin strokes
        cleaned_inner = self._remove_speckles(combined_inner, min_area=cfg["min_component_area"])

        # Fuse general lines + enhanced facial features
        if face_lines is not None and cv2.countNonZero(face_lines) > 0:
            cleaned_inner = cv2.bitwise_or(cleaned_inner, face_lines)

        # 8. Fuse Outer Silhouette + Internal Lines & apply anti-staircase smoothing
        final_lines = cv2.bitwise_or(outer_silhouette, cleaned_inner)
        final_lines = cv2.GaussianBlur(final_lines, (3, 3), 0)
        _, final_lines = cv2.threshold(final_lines, 105, 255, cv2.THRESH_BINARY)

        return final_lines

    @staticmethod
    def _remove_speckles(binary_lines: np.ndarray, min_area: int) -> np.ndarray:
        """Removes small disconnected noise dots while preserving facial and detail strokes."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_lines, connectivity=8)
        output = np.zeros_like(binary_lines)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            span = max(w, h)

            if area >= min_area or (area >= max(8, min_area // 2) and span >= 8):
                output[labels == i] = 255

        return output
