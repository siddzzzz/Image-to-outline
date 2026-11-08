from typing import Literal, Tuple
import cv2
import numpy as np

from .face_enhancer import FaceEnhancer
from .flow_field import EdgeTangentFlow

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
        self.flow_field = EdgeTangentFlow(iterations=1, radius=2)

    PRESETS = {
        "minimal": {
            "outer_thickness": 2,
            "canny_low": 30,
            "canny_high": 85,
            "dog_thresh": 6.5,
            "grad_thresh": 38,
            "min_span": 12,
            "min_area": 25,
        },
        "medium": {
            "outer_thickness": 2,
            "canny_low": 20,
            "canny_high": 60,
            "dog_thresh": 4.5,
            "grad_thresh": 28,
            "min_span": 8,
            "min_area": 14,
        },
        "high": {
            "outer_thickness": 2,
            "canny_low": 14,
            "canny_high": 45,
            "dog_thresh": 3.0,
            "grad_thresh": 18,
            "min_span": 5,
            "min_area": 8,
        },
    }

    def generate_line_bitmap(
        self,
        bgr_img: np.ndarray,
        fg_mask: np.ndarray,
        detail: DetailLevel = "medium",
        outer_line_thickness: int | None = None,
        enhance_faces: bool = True,
        use_flow_guidance: bool = True,
        return_flow: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # 4. Multi-channel Edge-Preserving Surface Smoothing & Flow-Guided Directional Filtering
        # Suppresses fabric/surface micro-noise while keeping all highlights, seams, and boundaries intact
        smooth_b = cv2.bilateralFilter(bgr_img[:, :, 0], 5, 25, 9)
        smooth_g = cv2.bilateralFilter(bgr_img[:, :, 1], 5, 25, 9)
        smooth_r = cv2.bilateralFilter(bgr_img[:, :, 2], 5, 25, 9)
        smooth_bgr = cv2.merge([smooth_b, smooth_g, smooth_r])

        tx, ty = None, None
        if use_flow_guidance:
            tx, ty, _ = self.flow_field.compute_field(bgr_img)
            # Smooth along edge tangents to eliminate perpendicular jitter and connect continuous ridges
            smooth_bgr = self.flow_field.filter_along_tangents(smooth_bgr, tx, ty, line_length=2)
            smooth_b, smooth_g, smooth_r = cv2.split(smooth_bgr)

        gray = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2GRAY)

        # 5. Multi-channel Morphological Color Gradient (captures colored highlights, graphics, badges, trims)
        k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gb = cv2.morphologyEx(smooth_b, cv2.MORPH_GRADIENT, k_grad)
        gg = cv2.morphologyEx(smooth_g, cv2.MORPH_GRADIENT, k_grad)
        gr = cv2.morphologyEx(smooth_r, cv2.MORPH_GRADIENT, k_grad)
        color_grad = np.maximum.reduce([gb, gg, gr])

        # 6. Symmetric Multi-Scale Difference of Gaussians (captures BOTH highlights/ridges AND dark creases/seams)
        g1 = cv2.GaussianBlur(gray, (0, 0), 0.75)
        g2 = cv2.GaussianBlur(gray, (0, 0), 2.1)
        dog = np.abs(g1.astype(np.float32) - g2.astype(np.float32))

        # 7. Multi-channel Sensitive Canny for crisp structural centerlines
        cb = cv2.Canny(smooth_b, cfg["canny_low"], cfg["canny_high"], L2gradient=True)
        cg = cv2.Canny(smooth_g, cfg["canny_low"], cfg["canny_high"], L2gradient=True)
        cr = cv2.Canny(smooth_r, cfg["canny_low"], cfg["canny_high"], L2gradient=True)
        color_canny = np.maximum.reduce([cb, cg, cr])

        # 8. Highlight ridges & high-contrast feature fusion
        ridge_mask = (dog > cfg["dog_thresh"]).astype(np.uint8) * 255
        high_contrast = (color_grad > cfg["grad_thresh"]).astype(np.uint8) * 255
        structural_lines = cv2.bitwise_or(color_canny, cv2.bitwise_and(ridge_mask, high_contrast))

        # 9. Clip to foreground mask
        structural_lines = cv2.bitwise_and(structural_lines, smooth_mask)

        # 10. Soft Skin Suppression on faces (only suppress low-contrast shadow noise on skin,
        # preserving high-contrast features like glasses frames, beards, and jewelry)
        if skin_mask is not None and cv2.countNonZero(skin_mask) > 0:
            low_contrast_skin = (skin_mask > 0) & (color_grad < 32)
            structural_lines[low_contrast_skin] = 0
            if face_lines is not None and cv2.countNonZero(face_lines) > 0:
                structural_lines = cv2.bitwise_or(structural_lines, face_lines)

        # 11. Bridge continuous strokes and prune isolated noise speckles
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bridged = cv2.dilate(structural_lines, k_bridge, iterations=1)
        cleaned_inner = self._remove_speckles(
            bridged, min_area=cfg["min_area"], min_span=cfg["min_span"]
        )

        # 12. Fuse Outer Silhouette + Internal Lines & apply anti-staircase smoothing
        final_lines = cv2.bitwise_or(outer_silhouette, cleaned_inner)
        final_lines = cv2.GaussianBlur(final_lines, (3, 3), 0)
        _, final_lines = cv2.threshold(final_lines, 90, 255, cv2.THRESH_BINARY)

        if return_flow:
            return final_lines, tx, ty
        return final_lines

    @staticmethod
    def _remove_speckles(binary_lines: np.ndarray, min_area: int, min_span: int = 8) -> np.ndarray:
        """Removes disconnected noise dots while preserving elongated strokes (laces, hair, seams)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_lines, connectivity=8)
        output = np.zeros_like(binary_lines)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            span = max(w, h)

            # Keep if elongated (stroke-like) OR if area meets threshold
            if span >= min_span or (area >= min_area and span >= max(4, min_span // 2)):
                output[labels == i] = 255

        return output
