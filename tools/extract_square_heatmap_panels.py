#!/usr/bin/env python3
"""Extract square, transparent heatmap objects and colorbars from comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, label


PANEL_SIZE = (340, 300)
PANEL_Y = 96
PANEL_LAYOUT = (
    ("physx3d", "affordance", 270),
    ("physx3d", "material", 620),
    ("physx_anything", "affordance", 970),
    ("physx_anything", "material", 1320),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("qualitative_heatmaps_mesh_aligned"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("qualitative_heatmaps_square_panels"),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--object-margin", type=int, default=34)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nonwhite_mask(rgb: np.ndarray, threshold: int = 246) -> np.ndarray:
    return np.min(rgb, axis=2) < threshold


def colored_object_mask(rgb: np.ndarray) -> np.ndarray:
    """Select PhysX-3D heatmap colors while rejecting gray axes and ticks."""
    value = np.max(rgb, axis=2).astype(np.float32)
    minimum = np.min(rgb, axis=2).astype(np.float32)
    saturation = (value - minimum) / np.maximum(value, 1.0)
    colored = (saturation > 0.20) & (value > 45.0)

    components, count = label(colored)
    filtered = np.zeros_like(colored)
    for component in range(1, count + 1):
        region = components == component
        if int(region.sum()) >= 12:
            filtered |= region
    if not np.any(filtered):
        return filtered

    # Retain genuinely black/gray low-valued regions when they touch a colored
    # object, without admitting the distant plot axes and tick labels.
    support = binary_fill_holes(binary_dilation(filtered, iterations=7))
    return filtered | (support & nonwhite_mask(rgb, 250))


def square_rgba(
    rgba: Image.Image,
    size: int,
    margin: int,
    resampling: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    array = np.asarray(rgba.convert("RGBA"))
    alpha = array[..., 3]
    coordinates = np.argwhere(alpha > 0)
    result = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    if not len(coordinates):
        return result
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    cropped = rgba.crop((int(left), int(top), int(right), int(bottom)))
    available = max(1, size - 2 * margin)
    scale = min(available / cropped.width, available / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        resampling,
    )
    result.alpha_composite(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return result


def extract_object(panel: Image.Image, method: str, size: int, margin: int) -> Image.Image:
    # Colorbars start around x=294.  Keep a safety gap so their outlines/ticks
    # cannot leak into the object crop.
    object_region = panel.convert("RGB").crop((0, 0, 280, PANEL_SIZE[1]))
    rgb = np.asarray(object_region)
    if method == "physx3d":
        alpha_mask = colored_object_mask(rgb)
    else:
        # CoVeTwin panels contain no axes inside the object region, so this keeps
        # black zero-valued parts as well as colored parts.
        alpha_mask = nonwhite_mask(rgb)
    rgba = np.dstack((rgb, (alpha_mask * 255).astype(np.uint8)))
    return square_rgba(Image.fromarray(rgba, "RGBA"), size, margin)


def extract_colorbar(panel: Image.Image, size: int) -> Image.Image:
    region = panel.convert("RGB").crop((282, 12, PANEL_SIZE[0], 286))
    rgb = np.asarray(region)
    alpha = (nonwhite_mask(rgb, 250) * 255).astype(np.uint8)
    rgba = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
    # A larger horizontal margin keeps the vertical bar visually distinct from
    # the square object thumbnail while retaining all tick labels.
    return square_rgba(rgba, size, margin=88)


def main() -> int:
    args = arguments()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    rows: list[dict[str, str]] = []
    generated = 0

    for variant in ("good", "bad"):
        for dataset in ("demo", "demo_new"):
            source_dir = input_root / variant / dataset
            if not source_dir.is_dir():
                continue
            for source in sorted(source_dir.glob("*.png"), key=lambda path: path.stem):
                with Image.open(source) as opened:
                    figure = opened.convert("RGB")
                if figure.width < 1660 or figure.height < PANEL_Y + PANEL_SIZE[1]:
                    print(f"[WARN] skip unexpected figure size {figure.size}: {source}")
                    continue
                for method, property_name, x in PANEL_LAYOUT:
                    destination = (
                        output_root
                        / variant
                        / dataset
                        / source.stem
                        / method
                        / property_name
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    heatmap_path = destination / "heatmap.png"
                    colorbar_path = destination / "colorbar.png"
                    if (
                        not args.overwrite
                        and heatmap_path.exists()
                        and colorbar_path.exists()
                    ):
                        status = "existing"
                    else:
                        panel = figure.crop(
                            (x, PANEL_Y, x + PANEL_SIZE[0], PANEL_Y + PANEL_SIZE[1])
                        )
                        extract_object(
                            panel, method, args.size, args.object_margin
                        ).save(heatmap_path)
                        extract_colorbar(panel, args.size).save(colorbar_path)
                        generated += 1
                        status = "generated"
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "sample": source.stem,
                            "method": method,
                            "property": property_name,
                            "status": status,
                            "heatmap": str(heatmap_path),
                            "colorbar": str(colorbar_path),
                        }
                    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "dataset",
                "sample",
                "method",
                "property",
                "status",
                "heatmap",
                "colorbar",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Panel pairs generated: {generated}")
    print(f"Manifest rows: {len(rows)}")
    print(f"Output: {output_root}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
