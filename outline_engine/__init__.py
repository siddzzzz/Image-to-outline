"""
Ultra-Lightweight Image-to-Outline Vector Engine.
Converts entities in images into smooth black-and-white Cubic Bézier SVG path equations.
"""

from .pipeline import ImageToOutlinePipeline, OutlineResult

__all__ = ["ImageToOutlinePipeline", "OutlineResult"]

