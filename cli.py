import argparse
from pathlib import Path
from outline_engine import ImageToOutlinePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Convert foreground entities in an image into compact Cubic Bézier SVG outlines."
    )
    parser.add_argument("input", type=Path, help="Path to input image file.")
    parser.add_argument(
        "-o", "--output-json", type=Path, default=None, help="Output path for compact DB JSON (.json)."
    )
    parser.add_argument(
        "-s", "--output-svg", type=Path, default=None, help="Output path for standalone SVG preview (.svg)."
    )
    parser.add_argument(
        "-d",
        "--detail",
        choices=["minimal", "medium", "high"],
        default="medium",
        help="Level of internal outline detail (default: medium).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=512,
        help="Maximum width/height for coordinate normalization (default: 512).",
    )
    parser.add_argument(
        "--keep-bg",
        action="store_true",
        help="Do not remove background before outlining.",
    )
    parser.add_argument(
        "--disable-face-enhance",
        action="store_true",
        help="Disable automatic facial landmark and feature enhancement.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    orig_size_kb = args.input.stat().st_size / 1024.0

    pipeline = ImageToOutlinePipeline(max_dimension=args.max_dim)
    res = pipeline.process(
        args.input,
        detail=args.detail,
        max_dimension=args.max_dim,
        remove_background=not args.keep_bg,
        enhance_faces=not args.disable_face_enhance,
    )

    json_path = args.output_json or args.input.with_suffix(".outline.json")
    svg_path = args.output_svg or args.input.with_suffix(".outline.svg")

    json_path.write_text(res.to_db_json(), encoding="utf-8")
    svg_path.write_text(res.svg_string, encoding="utf-8")

    json_kb = res.json_size_bytes / 1024.0
    gzip_kb = res.gzip_size_bytes / 1024.0
    savings_pct = (1.0 - (json_kb / max(orig_size_kb, 0.01))) * 100.0

    print(f"Done in {res.processing_ms:.1f} ms!")
    print(f"  Canvas:          {res.width}x{res.height} ({res.contours_count} curves, {res.anchor_points_count} anchor pts)")
    print(f"  Original Image:  {orig_size_kb:.1f} KB")
    print(f"  DB JSON Size:    {json_kb:.2f} KB ({gzip_kb:.2f} KB gzipped) -> {savings_pct:.1f}% smaller")
    print(f"  Saved DB JSON:   {json_path}")
    print(f"  Saved SVG File:  {svg_path}")


if __name__ == "__main__":
    main()

