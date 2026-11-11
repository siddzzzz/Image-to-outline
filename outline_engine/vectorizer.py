import gzip
import base64
import json
from typing import Literal
import numpy as np
import cv2

from .skeletonizer import StrokeSkeletonizer, CenterlineStroke
from .hierarchy import StrokeTier
from .curvature import AdaptiveBezierFitter

VectorMode = Literal["centerline", "tapered_ribbon", "ribbon"]


class BezierVectorizer:
    """
    Converts a binary line-art bitmap into compact mathematical Cubic Bézier SVG Path
    equations (M, L, C, Z) optimized for minimal database storage and smooth rendering.

    Supported Modes:
    1. 'centerline': Extracts 1-pixel centerlines with stroke width and endpoint tapering.
       Ultra-compact (~50% smaller DB payload), and animatable with SVG stroke-dashoffset.
    2. 'tapered_ribbon': Extrudes centerlines into brush-pen tapered closed ribbons.
    3. 'ribbon': Traditional filled contour ribbon boundaries.
    """

    def __init__(
        self,
        simplify_epsilon: float = 0.65,
        curve_tension: float = 0.45,
        corner_angle_deg: float = 55.0,
        precision: int = 0,
        min_contour_area: float = 3.0,
        mode: VectorMode = "centerline",
    ):
        self.simplify_epsilon = simplify_epsilon
        self.curve_tension = curve_tension
        self.corner_cos_thresh = float(np.cos(np.radians(corner_angle_deg)))
        self.precision = precision
        self.min_contour_area = min_contour_area
        self.mode = mode
        self.skeletonizer = StrokeSkeletonizer()
        self.fitter = AdaptiveBezierFitter(
            max_error=max(0.6, self.simplify_epsilon * 1.2),
            corner_thresh_deg=corner_angle_deg,
            collinear_thresh=0.65,
            precision=self.precision,
        )

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
        Applies a gentle closed-loop filter to remove 1px raster stair-stepping.
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
        Fits a closed sequence of 2D anchor points with adaptive Cubic Bézier curves,
        preserving sharp corner cusps and reducing straight lines to compact 'L' commands.
        Ends with 'Z'.
        """
        return self.fitter.fit_stroke_to_path(pts, is_closed=True)

    def _open_stroke_to_bezier_path(self, pts: np.ndarray) -> str:
        """
        Fits an open sequence of 2D anchor points with adaptive Cubic Bézier curves,
        preserving sharp corner cusps and reducing straight lines to compact 'L' commands.
        Does not close with 'Z'.
        """
        return self.fitter.fit_stroke_to_path(pts, is_closed=False)

        return "".join(cmds)

    def trace_centerlines(
        self,
        binary_bitmap: np.ndarray,
        tx: np.ndarray | None = None,
        ty: np.ndarray | None = None,
        silhouette_mask: np.ndarray | None = None,
        structure_mask: np.ndarray | None = None,
        mag_map: np.ndarray | None = None,
        min_tier: StrokeTier = StrokeTier.DETAIL,
        max_detail_strokes: int | None = None,
    ) -> dict:
        """
        Vectorizes 1-pixel centerlines into open/closed Cubic Bézier stroke paths.
        Yields ultra-compact JSON representation (~50% smaller than ribbon mode),
        with strokes tagged and ordered by structural hierarchy.
        """
        h, w = binary_bitmap.shape[:2]
        strokes = self.skeletonizer.trace_skeleton_strokes(
            binary_bitmap,
            min_length=4,
            tx=tx,
            ty=ty,
            silhouette_mask=silhouette_mask,
            structure_mask=structure_mask,
            mag_map=mag_map,
        )

        if min_tier < StrokeTier.DETAIL or max_detail_strokes is not None:
            strokes = self.skeletonizer.classifier.prune_strokes(
                strokes,
                min_tier=min_tier,
                max_detail_strokes=max_detail_strokes,
            )

        if not strokes:
            return {
                "w": w,
                "h": h,
                "d": "",
                "mode": "centerline",
                "stroke_width": 1.8,
                "contours": 0,
                "points": 0,
                "tiers": {"silhouette": "", "structure": "", "detail": ""},
                "tier_counts": {"silhouette": 0, "structure": 0, "detail": 0},
            }

        tier_segments: dict[StrokeTier, list[str]] = {
            StrokeTier.SILHOUETTE: [],
            StrokeTier.STRUCTURE: [],
            StrokeTier.DETAIL: [],
        }
        total_points = 0
        all_radii = []

        for stroke in strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue

            if stroke.is_closed:
                subpath = self._contour_to_bezier_path(pts)
            else:
                subpath = self._open_stroke_to_bezier_path(pts)

            if subpath:
                tier_segments[stroke.tier].append(subpath)
                total_points += len(pts)
                all_radii.append(float(np.median(stroke.radii)))

        avg_stroke_w = max(1.2, float(np.median(all_radii) * 2.0)) if all_radii else 1.8
        path_segments = (
            tier_segments[StrokeTier.SILHOUETTE]
            + tier_segments[StrokeTier.STRUCTURE]
            + tier_segments[StrokeTier.DETAIL]
        )

        return {
            "w": w,
            "h": h,
            "d": "".join(path_segments),
            "mode": "centerline",
            "stroke_width": round(avg_stroke_w, 1),
            "contours": len(path_segments),
            "points": total_points,
            "tiers": {
                "silhouette": "".join(tier_segments[StrokeTier.SILHOUETTE]),
                "structure": "".join(tier_segments[StrokeTier.STRUCTURE]),
                "detail": "".join(tier_segments[StrokeTier.DETAIL]),
            },
            "tier_counts": {
                "silhouette": len(tier_segments[StrokeTier.SILHOUETTE]),
                "structure": len(tier_segments[StrokeTier.STRUCTURE]),
                "detail": len(tier_segments[StrokeTier.DETAIL]),
            },
        }

    def trace_tapered_ribbons(
        self,
        binary_bitmap: np.ndarray,
        tx: np.ndarray | None = None,
        ty: np.ndarray | None = None,
        silhouette_mask: np.ndarray | None = None,
        structure_mask: np.ndarray | None = None,
        mag_map: np.ndarray | None = None,
        min_tier: StrokeTier = StrokeTier.DETAIL,
        max_detail_strokes: int | None = None,
    ) -> dict:
        """
        Vectorizes brush-pen tapered ribbons with calligraphy pressure variation at stroke tips.
        """
        h, w = binary_bitmap.shape[:2]
        strokes = self.skeletonizer.trace_skeleton_strokes(
            binary_bitmap,
            min_length=4,
            tx=tx,
            ty=ty,
            silhouette_mask=silhouette_mask,
            structure_mask=structure_mask,
            mag_map=mag_map,
        )

        if min_tier < StrokeTier.DETAIL or max_detail_strokes is not None:
            strokes = self.skeletonizer.classifier.prune_strokes(
                strokes,
                min_tier=min_tier,
                max_detail_strokes=max_detail_strokes,
            )

        if not strokes:
            return {
                "w": w,
                "h": h,
                "d": "",
                "mode": "tapered_ribbon",
                "stroke_width": None,
                "contours": 0,
                "points": 0,
                "tiers": {"silhouette": "", "structure": "", "detail": ""},
                "tier_counts": {"silhouette": 0, "structure": 0, "detail": 0},
            }

        tier_segments: dict[StrokeTier, list[str]] = {
            StrokeTier.SILHOUETTE: [],
            StrokeTier.STRUCTURE: [],
            StrokeTier.DETAIL: [],
        }
        total_points = 0

        for stroke in strokes:
            ribbon_pts = stroke.extrude_tapered_ribbon(taper_fraction=0.25)
            if len(ribbon_pts) < 3:
                continue

            subpath = self._contour_to_bezier_path(ribbon_pts)
            if subpath:
                tier_segments[stroke.tier].append(subpath)
                total_points += len(ribbon_pts)

        path_segments = (
            tier_segments[StrokeTier.SILHOUETTE]
            + tier_segments[StrokeTier.STRUCTURE]
            + tier_segments[StrokeTier.DETAIL]
        )

        return {
            "w": w,
            "h": h,
            "d": "".join(path_segments),
            "mode": "tapered_ribbon",
            "stroke_width": None,
            "contours": len(path_segments),
            "points": total_points,
            "tiers": {
                "silhouette": "".join(tier_segments[StrokeTier.SILHOUETTE]),
                "structure": "".join(tier_segments[StrokeTier.STRUCTURE]),
                "detail": "".join(tier_segments[StrokeTier.DETAIL]),
            },
            "tier_counts": {
                "silhouette": len(tier_segments[StrokeTier.SILHOUETTE]),
                "structure": len(tier_segments[StrokeTier.STRUCTURE]),
                "detail": len(tier_segments[StrokeTier.DETAIL]),
            },
        }

    def trace_bitmap(
        self,
        binary_bitmap: np.ndarray,
        mode: VectorMode | None = None,
        tx: np.ndarray | None = None,
        ty: np.ndarray | None = None,
        silhouette_mask: np.ndarray | None = None,
        structure_mask: np.ndarray | None = None,
        mag_map: np.ndarray | None = None,
        min_tier: StrokeTier = StrokeTier.DETAIL,
        max_detail_strokes: int | None = None,
    ) -> dict:
        """
        Main vectorization entry point. Selects mode ('centerline', 'tapered_ribbon', or 'ribbon').
        """
        selected_mode = mode or self.mode
        if selected_mode == "centerline":
            return self.trace_centerlines(
                binary_bitmap,
                tx=tx,
                ty=ty,
                silhouette_mask=silhouette_mask,
                structure_mask=structure_mask,
                mag_map=mag_map,
                min_tier=min_tier,
                max_detail_strokes=max_detail_strokes,
            )
        elif selected_mode == "tapered_ribbon":
            return self.trace_tapered_ribbons(
                binary_bitmap,
                tx=tx,
                ty=ty,
                silhouette_mask=silhouette_mask,
                structure_mask=structure_mask,
                mag_map=mag_map,
                min_tier=min_tier,
                max_detail_strokes=max_detail_strokes,
            )

        # Traditional ribbon mode (contour boundaries)
        h, w = binary_bitmap.shape[:2]
        contours, hierarchy = cv2.findContours(
            binary_bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours or hierarchy is None:
            return {
                "w": w,
                "h": h,
                "d": "",
                "mode": "ribbon",
                "stroke_width": None,
                "contours": 0,
                "points": 0,
                "tiers": {"silhouette": "", "structure": "", "detail": ""},
                "tier_counts": {"silhouette": 0, "structure": 0, "detail": 0},
            }

        hier = hierarchy[0]
        path_segments: list[str] = []
        total_points = 0

        for idx, raw_cnt in enumerate(contours):
            area = abs(cv2.contourArea(raw_cnt))
            if area < self.min_contour_area and len(raw_cnt) < 6:
                continue

            pts = raw_cnt[:, 0, :]
            is_hole = hier[idx][3] != -1
            signed_area = cv2.contourArea(pts.reshape(-1, 1, 2), oriented=True)
            if (not is_hole and signed_area < 0) or (is_hole and signed_area > 0):
                pts = pts[::-1]

            subpath = self._contour_to_bezier_path(pts)
            if subpath:
                path_segments.append(subpath)
                total_points += len(pts)

        return {
            "w": w,
            "h": h,
            "d": "".join(path_segments),
            "mode": "ribbon",
            "stroke_width": None,
            "contours": len(path_segments),
            "points": total_points,
            "tiers": {"silhouette": "".join(path_segments), "structure": "", "detail": ""},
            "tier_counts": {"silhouette": len(path_segments), "structure": 0, "detail": 0},
        }

    @staticmethod
    def to_svg_string(
        path_data: dict,
        stroke_color: str = "#000000",
        background_color: str | None = "#ffffff",
        hierarchical_weights: bool = False,
    ) -> str:
        """
        Wraps path data into a valid standalone SVG document based on vector mode.
        If hierarchical_weights is True, renders multi-tier line weights (bold silhouette,
        medium structure, fine details) with semantic <g> tier grouping.
        """
        w, h, d = path_data["w"], path_data["h"], path_data["d"]
        mode = path_data.get("mode", "ribbon")
        sw = path_data.get("stroke_width", 1.8) or 1.8

        bg_rect = (
            f'<rect width="100%" height="100%" fill="{background_color}"/>\n'
            if background_color
            else ""
        )

        tiers = path_data.get("tiers")

        if mode == "centerline":
            if hierarchical_weights and tiers:
                sil_d = tiers.get("silhouette", "")
                str_d = tiers.get("structure", "")
                det_d = tiers.get("detail", "")
                sw_sil = round(sw * 1.5, 1)
                sw_str = round(sw * 1.0, 1)
                sw_det = round(max(0.9, sw * 0.65), 1)

                elements = []
                if sil_d:
                    elements.append(
                        f'  <g id="tier-silhouette"><path d="{sil_d}" stroke="{stroke_color}" stroke-width="{sw_sil}" stroke-linecap="round" stroke-linejoin="round" fill="none"/></g>'
                    )
                if str_d:
                    elements.append(
                        f'  <g id="tier-structure"><path d="{str_d}" stroke="{stroke_color}" stroke-width="{sw_str}" stroke-linecap="round" stroke-linejoin="round" fill="none"/></g>'
                    )
                if det_d:
                    elements.append(
                        f'  <g id="tier-detail"><path d="{det_d}" stroke="{stroke_color}" stroke-width="{sw_det}" stroke-linecap="round" stroke-linejoin="round" fill="none"/></g>'
                    )
                content = "\n".join(elements)
            else:
                content = (
                    f'  <path d="{d}" stroke="{stroke_color}" stroke-width="{sw}" '
                    f'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
                )

            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                f'width="{w}" height="{h}">\n'
                f'  {bg_rect}'
                f'{content}\n'
                f'</svg>'
            )
        else:
            if hierarchical_weights and tiers:
                sil_d = tiers.get("silhouette", "")
                str_d = tiers.get("structure", "")
                det_d = tiers.get("detail", "")

                elements = []
                if sil_d:
                    elements.append(f'  <g id="tier-silhouette"><path d="{sil_d}" fill="{stroke_color}" fill-rule="evenodd"/></g>')
                if str_d:
                    elements.append(f'  <g id="tier-structure"><path d="{str_d}" fill="{stroke_color}" fill-rule="evenodd"/></g>')
                if det_d:
                    elements.append(f'  <g id="tier-detail"><path d="{det_d}" fill="{stroke_color}" fill-rule="evenodd"/></g>')
                content = "\n".join(elements)
            else:
                content = f'  <path d="{d}" fill="{stroke_color}" fill-rule="evenodd"/>'

            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                f'width="{w}" height="{h}">\n'
                f'  {bg_rect}'
                f'{content}\n'
                f'</svg>'
            )

    @staticmethod
    def compress_payload(db_payload: dict) -> str:
        raw_bytes = json.dumps(db_payload, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw_bytes, compresslevel=9)
        return base64.b64encode(compressed).decode("ascii")
