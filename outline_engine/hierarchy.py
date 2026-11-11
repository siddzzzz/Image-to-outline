"""
Hierarchical Multi-Scale Stroke Classification and Level-of-Detail (LOD) Pruning.

Classifies extracted vector strokes into three semantic structural tiers:
1. SILHOUETTE (Tier 0): The outer entity boundary separating subject from background.
2. STRUCTURE  (Tier 1): Major interior contours (facial features, collar, arms, rims).
3. DETAIL     (Tier 2): Secondary folds, hair strands, fabric wrinkles, textures.

Enables:
- Per-tier line weight rendering (e.g. bold silhouette, medium structure, fine detail).
- Smart hierarchy pruning that drops low-salience details to hit a strict database byte budget
  without ever degrading the essential silhouette or facial structure.
"""

from enum import IntEnum
from typing import Sequence
import numpy as np
import cv2


class StrokeTier(IntEnum):
    """Semantic hierarchy level for outline strokes."""
    SILHOUETTE = 0  # Primary outer subject boundary (bold line weight)
    STRUCTURE = 1   # Major interior anatomy/creases/landmarks (medium line weight)
    DETAIL = 2      # Fine secondary wrinkles, hair strands, texture (fine line weight)


class HierarchyClassifier:
    """
    Classifies vector strokes into semantic hierarchy tiers and provides
    salience-driven pruning algorithms for dynamic Level-of-Detail (LOD).
    """

    def __init__(
        self,
        silhouette_overlap_thresh: float = 0.35,
        structure_overlap_thresh: float = 0.25,
        high_gradient_thresh: float = 65.0,
    ):
        self.silhouette_overlap_thresh = silhouette_overlap_thresh
        self.structure_overlap_thresh = structure_overlap_thresh
        self.high_gradient_thresh = high_gradient_thresh

    def classify_stroke(
        self,
        points: np.ndarray,
        silhouette_mask: np.ndarray | None,
        structure_mask: np.ndarray | None = None,
        mag_map: np.ndarray | None = None,
    ) -> tuple[StrokeTier, float]:
        """
        Classifies a single stroke (N, 2) into a StrokeTier and computes a salience score.

        Returns:
            tier: StrokeTier (SILHOUETTE, STRUCTURE, or DETAIL).
            salience: Positive float score measuring the stroke's visual importance.
        """
        n_pts = len(points)
        if n_pts == 0:
            return StrokeTier.DETAIL, 0.0

        h, w = silhouette_mask.shape[:2] if silhouette_mask is not None else (512, 512)

        int_pts = np.round(points).astype(np.int32)
        xs = np.clip(int_pts[:, 0], 0, w - 1)
        ys = np.clip(int_pts[:, 1], 0, h - 1)

        # 1. Check overlap with outer silhouette boundary
        if silhouette_mask is not None:
            sil_hits = int(np.count_nonzero(silhouette_mask[ys, xs]))
            if (sil_hits / float(n_pts)) >= self.silhouette_overlap_thresh:
                salience = 1000.0 + float(n_pts)
                return StrokeTier.SILHOUETTE, salience

        # 2. Check overlap with structure mask (e.g. facial features, key seams)
        mean_mag = 0.0
        if mag_map is not None:
            mean_mag = float(np.mean(mag_map[ys, xs]))

        if structure_mask is not None:
            struct_hits = int(np.count_nonzero(structure_mask[ys, xs]))
            if (struct_hits / float(n_pts)) >= self.structure_overlap_thresh:
                salience = 500.0 + mean_mag + float(n_pts)
                return StrokeTier.STRUCTURE, salience

        # High-contrast interior ridges also qualify as major structure
        if mean_mag >= self.high_gradient_thresh and n_pts >= 10:
            salience = 300.0 + mean_mag + float(n_pts)
            return StrokeTier.STRUCTURE, salience

        # 3. Secondary detail strokes
        salience = mean_mag + float(n_pts)
        return StrokeTier.DETAIL, salience

    def prune_strokes(
        self,
        strokes: Sequence,
        min_tier: StrokeTier = StrokeTier.DETAIL,
        min_length_detail: int = 5,
        max_detail_strokes: int | None = None,
    ) -> list:
        """
        Prunes strokes based on structural hierarchy and detail salience:
        - If min_tier == SILHOUETTE: retains only silhouette strokes.
        - If min_tier == STRUCTURE: retains silhouette and structural strokes.
        - If min_tier == DETAIL: retains all tiers, filtering tiny noise specks in detail tier.
        - If max_detail_strokes is set: greedily keeps the top-N most salient detail strokes.
        """
        kept_sil = []
        kept_struct = []
        detail_candidates = []

        for stroke in strokes:
            tier = getattr(stroke, "tier", StrokeTier.DETAIL)
            salience = getattr(stroke, "salience", float(len(stroke.points)))

            if tier == StrokeTier.SILHOUETTE:
                kept_sil.append(stroke)
            elif tier == StrokeTier.STRUCTURE:
                if min_tier >= StrokeTier.STRUCTURE:
                    kept_struct.append(stroke)
            else:  # DETAIL
                if min_tier >= StrokeTier.DETAIL:
                    # Filter out tiny isolated noise strokes in the detail tier
                    if len(stroke.points) >= min_length_detail:
                        detail_candidates.append((salience, stroke))

        # Sort detail strokes by salience descending
        detail_candidates.sort(key=lambda x: x[0], reverse=True)
        if max_detail_strokes is not None and max_detail_strokes >= 0:
            detail_candidates = detail_candidates[:max_detail_strokes]

        kept_details = [stroke for _, stroke in detail_candidates]
        return kept_sil + kept_struct + kept_details
