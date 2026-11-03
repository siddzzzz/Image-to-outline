import gzip
import base64
import json
import numpy as np
import cv2


class BezierVectorizer:
    """
    Converts a binary line-art bitmap into compact mathematical Cubic Bézier SVG Path
    equations (M, L, C, Z) optimized for minimal database storage and high-precision
    facial/detail rendering.
    """

    def __init__(
        self,
        simplify_epsilon: float = 0.65,
        curve_tension: float = 0.45,
        corner_angle_deg: float = 55.0,
        precision: int = 0,
        min_contour_area: float = 3.0,
    ):
        self.simplify_epsilon = simplify_epsilon
        self.curve_tension = curve_tension
        self.corner_cos_thresh = float(np.cos(np.radians(corner_angle_deg)))
        self.precision = precision
        self.min_contour_area = min_contour_area

    def _fmt(self, val: float) -> str:
        if self.precision <= 0:
            return str(int(round(val)))
        rounded = round(val, self.precision)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{self.precision}f}"

    @staticmethod
    def _smooth_pixel_chain(pts: np.ndarray) -> np.ndarray:
        """
        Applies a single gentle closed-loop filter to remove 1px raster stair-stepping
        without shrinking small facial features (eyes, mouths) in group photos.
        """
        n = len(pts)
        if n < 6:
            return pts.astype(np.float32)

        curr = pts.astype(np.float32)
        prev_pts = np.roll(curr, 1, axis=0)
        next_pts = np.roll(curr, -1, axis=0)
        return 0.18 * prev_pts + 0.64 * curr + 0.18 * next_pts

    def _contour_to_bezier_path(self, pts: np.ndarray) -> str:
        """
        Fits a closed sequence of 2D anchor points with piecewise Cubic Bézier curves
        (C) for smooth arcs and linear segments (L) for short/straight edges.
        """
        n = len(pts)
        if n < 3:
            return ""

        if n <= 4:
            coords = "L".join(f"{self._fmt(p[0])} {self._fmt(p[1])}" for p in pts[1:])
            return f"M{self._fmt(pts[0][0])} {self._fmt(pts[0][1])}L{coords}Z"

        cmds = [f"M{self._fmt(pts[0][0])} {self._fmt(pts[0][1])}"]
        tension_scale = self.curve_tension / 3.0

        seg_vecs = np.roll(pts, -1, axis=0) - pts
        seg_lens = np.linalg.norm(seg_vecs, axis=1, keepdims=True) + 1e-8
        unit_vecs = seg_vecs / seg_lens

        in_vecs = np.roll(unit_vecs, 1, axis=0)
        out_vecs = unit_vecs
        cos_angles = np.sum(in_vecs * out_vecs, axis=1)
        is_corner = cos_angles < self.corner_cos_thresh

        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]

            seg_len = float(seg_lens[i, 0])
            if seg_len < 3.5 or (is_corner[i] and is_corner[(i + 1) % n]):
                cmds.append(f"L{self._fmt(p2[0])} {self._fmt(p2[1])}")
                continue

            if is_corner[i]:
                c1 = p1 + (p2 - p1) * 0.25
            else:
                t1 = (p2 - p0) * tension_scale
                t1_len = float(np.linalg.norm(t1))
                if t1_len > seg_len * 0.42:
                    t1 = t1 * (seg_len * 0.42 / (t1_len + 1e-8))
                c1 = p1 + t1

            if is_corner[(i + 1) % n]:
                c2 = p2 - (p2 - p1) * 0.25
            else:
                t2 = (p3 - p1) * tension_scale
                t2_len = float(np.linalg.norm(t2))
                if t2_len > seg_len * 0.42:
                    t2 = t2 * (seg_len * 0.42 / (t2_len + 1e-8))
                c2 = p2 - t2

            cmds.append(
                f"C{self._fmt(c1[0])} {self._fmt(c1[1])} "
                f"{self._fmt(c2[0])} {self._fmt(c2[1])} "
                f"{self._fmt(p2[0])} {self._fmt(p2[1])}"
            )

        cmds.append("Z")
        return "".join(cmds)

    def trace_bitmap(self, binary_bitmap: np.ndarray) -> dict:
        """
        Traces a binary uint8 bitmap (255 = black ink, 0 = white background)
        into a compact SVG path dictionary ready for database storage.
        """
        h, w = binary_bitmap.shape[:2]

        contours, hierarchy = cv2.findContours(
            binary_bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours or hierarchy is None:
            return {
                "w": w,
                "h": h,
                "d": "",
                "contours": 0,
                "points": 0,
            }

        hier = hierarchy[0]
        path_segments: list[str] = []
        total_points = 0

        for idx, raw_cnt in enumerate(contours):
            area = abs(cv2.contourArea(raw_cnt))
            if area < self.min_contour_area and len(raw_cnt) < 6:
                continue

            pts = raw_cnt[:, 0, :]
            smoothed = self._smooth_pixel_chain(pts)

            approx = cv2.approxPolyDP(
                smoothed.reshape(-1, 1, 2),
                epsilon=self.simplify_epsilon,
                closed=True,
            )[:, 0, :]

            if len(approx) < 3:
                continue

            is_hole = hier[idx][3] != -1
            signed_area = cv2.contourArea(approx.reshape(-1, 1, 2), oriented=True)
            if (not is_hole and signed_area < 0) or (is_hole and signed_area > 0):
                approx = approx[::-1]

            subpath = self._contour_to_bezier_path(approx)
            if subpath:
                path_segments.append(subpath)
                total_points += len(approx)

        compound_path = "".join(path_segments)
        return {
            "w": w,
            "h": h,
            "d": compound_path,
            "contours": len(path_segments),
            "points": total_points,
        }

    @staticmethod
    def to_svg_string(
        path_data: dict,
        stroke_color: str = "#000000",
        background_color: str | None = "#ffffff",
    ) -> str:
        w, h, d = path_data["w"], path_data["h"], path_data["d"]
        bg_rect = (
            f'<rect width="100%" height="100%" fill="{background_color}"/>'
            if background_color
            else ""
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">'
            f'{bg_rect}'
            f'<path d="{d}" fill="{stroke_color}" fill-rule="evenodd"/>'
            f"</svg>"
        )

    @staticmethod
    def compress_payload(db_payload: dict) -> str:
        raw_bytes = json.dumps(db_payload, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw_bytes, compresslevel=9)
        return base64.b64encode(compressed).decode("ascii")
