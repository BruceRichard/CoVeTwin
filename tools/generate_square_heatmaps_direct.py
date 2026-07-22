#!/usr/bin/env python3
"""Directly generate square heatmaps from URDF meshes and PhysX-3D source videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluate_metrics as metric
import make_qualitative_heatmaps_mesh as vis


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physx3d-root",
        type=Path,
        default=Path("/mnt/data/zhangzhaodong/PhysX-3D"),
    )
    parser.add_argument(
        "--anything-root",
        type=Path,
        default=Path("/mnt/data/zhangzhaodong/PhysX-Anything"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("qualitative_heatmaps_square_direct"),
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(vis.DATASETS), default=list(vis.DATASETS))
    parser.add_argument("--variants", nargs="+", choices=("good", "bad"), default=["good", "bad"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--supersampling", type=int, default=2)
    parser.add_argument("--alignment-samples", type=int, default=4000)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("/mnt/data/zhangzhaodong/PhysX-Anything/Ablation"),
    )
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="Also generate every Ablation/test_* CoVeTwin variant.",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Generate Ablation/test_* variants without regenerating main methods.",
    )
    parser.add_argument(
        "--refresh-physx3d-colorbars",
        action="store_true",
        help="Only rebuild PhysX-3D colorbars from their source MP4 numeric scales.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def alpha_square(image: Image.Image, size: int, margin: int = 26) -> Image.Image:
    rgba = image.convert("RGBA")
    array = np.asarray(rgba)
    coordinates = np.argwhere(array[..., 3] > 0)
    result = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    if not len(coordinates):
        return result
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    cropped = rgba.crop((int(left), int(top), int(right), int(bottom)))
    available = size - 2 * margin
    scale = min(available / cropped.width, available / cropped.height)
    fitted = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result.alpha_composite(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return result


def direct_twinx_heatmap(
    part_ids: np.ndarray,
    values: dict[int, float],
    size: int,
) -> Image.Image:
    rgb = np.zeros((*part_ids.shape, 3), dtype=np.uint8)
    alpha = np.zeros(part_ids.shape, dtype=np.uint8)
    cmap = colormaps["jet"]
    for encoded_label in np.unique(part_ids):
        if encoded_label <= 0:
            continue
        label = int(encoded_label) - 1
        value = float(np.clip(values.get(label, 0.0), 0.0, 1.0))
        # Use the actual jet endpoint for zero instead of pure black.  Pure
        # black becomes invisible in viewers that display transparency on a
        # black canvas and is inconsistent with the accompanying colorbar.
        color = tuple(int(channel * 255) for channel in cmap(value)[:3])
        selected = part_ids == encoded_label
        rgb[selected] = color
        alpha[selected] = 255
    rgba = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
    if rgba.size != (size, size):
        rgba = rgba.resize((size, size), Image.Resampling.LANCZOS)
    return rgba


def physx3d_source_heatmap(
    heatmap_frame: Image.Image,
    rgb_frame: Image.Image,
    size: int,
) -> Image.Image:
    """Recover the source plot using its RGB-frame silhouette, not a composite crop."""
    heat = np.asarray(heatmap_frame.convert("RGB"), dtype=np.uint8)
    black = np.max(heat, axis=2) < 18
    rows = np.where(black.mean(axis=1) > 0.25)[0]
    columns = np.where(black.mean(axis=0) > 0.25)[0]
    if not len(rows) or not len(columns):
        return Image.new("RGBA", (size, size), (255, 255, 255, 0))
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    plot = heat[top:bottom, left:right]

    rgb = np.asarray(rgb_frame.convert("RGB"), dtype=np.uint8)
    foreground = (np.max(rgb, axis=2) > 8).astype(np.uint8) * 255
    alpha = np.asarray(
        Image.fromarray(foreground, "L").resize(
            (right - left, bottom - top), Image.Resampling.LANCZOS
        )
    )
    return alpha_square(Image.fromarray(np.dstack((plot, alpha)), "RGBA"), size)


def clean_colorbar(
    size: int,
    ticks: list[tuple[float, str]],
    title: str,
) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    cmap = colormaps["jet"]
    left, right = round(size * 0.39), round(size * 0.49)
    top, bottom = round(size * 0.10), round(size * 0.88)
    for y in range(top, bottom):
        value = 1.0 - (y - top) / max(1, bottom - top - 1)
        color = tuple(int(channel * 255) for channel in cmap(value)[:3]) + (255,)
        draw.line((left, y, right, y), fill=color, width=1)
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0, 255), width=2)
    font = vis.load_font(max(18, round(size * 0.045)))
    title_font = vis.load_font(max(19, round(size * 0.047)), bold=True)
    for position, label in ticks:
        y = round(bottom - position * (bottom - top))
        draw.line((right + 2, y, right + 13, y), fill=(0, 0, 0, 255), width=2)
        draw.text((right + 18, y - font.size // 2), label, font=font, fill=(0, 0, 0, 255))
    bounds = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((size - (bounds[2] - bounds[0])) / 2, round(size * 0.025)),
        title,
        font=title_font,
        fill=(0, 0, 0, 255),
    )
    return image


def source_numeric_colorbar(frame: Image.Image, size: int, title: str) -> Image.Image:
    """Extract the source method's exact numeric bar into a transparent square."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    # PhysX-3D's matplotlib output places its bar and all three numeric labels
    # in the rightmost 16% of the frame.  Keeping the source labels preserves
    # the per-object min/mid/max scale instead of replacing it with categories.
    left = round(width * 0.84)
    top = round(height * 0.07)
    bottom = round(height * 0.94)
    crop = rgb[top:bottom, left:width]
    nonwhite = np.min(crop, axis=2) < 248
    coordinates = np.argwhere(nonwhite)
    result = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    if not len(coordinates):
        return result
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    crop = crop[y0:y1, x0:x1]
    # A hard threshold removes faint H.264 ringing around tick labels.  A
    # continuous alpha would turn those almost-white compression artifacts
    # into visible duplicated/ghosted text on transparent backgrounds.
    alpha = np.where(np.min(crop, axis=2) < 225, 255, 0).astype(np.uint8)
    rgba = Image.fromarray(np.dstack((crop, alpha)), "RGBA")
    available_width = round(size * 0.52)
    available_height = round(size * 0.77)
    scale = min(available_width / rgba.width, available_height / rgba.height)
    fitted = rgba.resize(
        (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result.alpha_composite(fitted, ((size - fitted.width) // 2, round(size * 0.12)))
    draw = ImageDraw.Draw(result)
    title_font = vis.load_font(max(19, round(size * 0.047)), bold=True)
    bounds = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((size - (bounds[2] - bounds[0])) / 2, round(size * 0.025)),
        title,
        font=title_font,
        fill=(0, 0, 0, 255),
    )
    return result


def refresh_physx3d_colorbars(
    physx3d_root: Path,
    output_root: Path,
    size: int,
) -> int:
    manifest = output_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing main manifest: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("method") == "physx3d"]
    cache: dict[tuple[Path, int], Image.Image] = {}
    rebuilt = 0
    with tempfile.TemporaryDirectory(prefix="physx3d_bars_") as temporary:
        temporary_root = Path(temporary)
        for row in rows:
            dataset_name = row["dataset"]
            stem = row["sample"]
            property_name = row["property"]
            spec = vis.DATASETS[dataset_name]
            sample_dir = vis.find_physx3d_sample(physx3d_root, spec.physx3d_dirs, stem)
            if sample_dir is None:
                continue
            video = sample_dir / f"{property_name}.mp4"
            frame_number = int(row["frame"])
            key = (video, frame_number)
            if key not in cache:
                cache[key] = vis.extract_video_frame(
                    video,
                    frame_number,
                    temporary_root / f"{dataset_name}_{stem}_{property_name}_{frame_number}.png",
                )
            bar = source_numeric_colorbar(cache[key], size, property_name.title())
            target = Path(row["colorbar"])
            target.parent.mkdir(parents=True, exist_ok=True)
            bar.save(target)
            rebuilt += 1
    print(f"Rebuilt PhysX-3D numeric colorbars: {rebuilt}")
    return rebuilt


def save_pair(
    directory: Path,
    heatmap: Image.Image,
    colorbar: Image.Image,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    heatmap_path = directory / "heatmap.png"
    colorbar_path = directory / "colorbar.png"
    heatmap.save(heatmap_path)
    colorbar.save(colorbar_path)
    return heatmap_path, colorbar_path


def main() -> int:
    args = arguments()
    physx3d_root = args.physx3d_root.resolve()
    anything_root = args.anything_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.refresh_physx3d_colorbars:
        refresh_physx3d_colorbars(physx3d_root, output_root, args.size)
        return 0
    selected_ids = set(args.only)
    view_cache: dict[tuple[Path, Path], tuple[int, float]] = {}
    orientation_cache: dict[str, tuple[np.ndarray, int, float]] = {}
    extracted_cache: dict[tuple[Path, int], Image.Image] = {}
    rows: list[dict[str, object]] = []

    alignment_args = argparse.Namespace(
        mesh_alignment="input_view",
        alignment_samples=args.alignment_samples,
        icp_iterations=0,
        icp_candidates=0,
    )

    with tempfile.TemporaryDirectory(prefix="direct_heatmaps_") as temporary:
        temporary_root = Path(temporary)
        main_datasets = [] if args.ablation_only else args.datasets
        for dataset_name in main_datasets:
            spec = vis.DATASETS[dataset_name]
            input_dir = physx3d_root / spec.input_dir
            stems = sorted(
                {path.stem for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in vis.IMAGE_EXTENSIONS},
                key=vis.natural_key,
            )
            if selected_ids:
                stems = [stem for stem in stems if stem in selected_ids]
            for variant in args.variants:
                anything_name = spec.anything_good if variant == "good" else spec.anything_bad
                for index, stem in enumerate(stems, start=1):
                    sample_key = f"{dataset_name}/{stem}"
                    print(f"[{variant} {dataset_name} {index}/{len(stems)}] {stem}", flush=True)
                    input_path = vis.find_input(input_dir, stem)
                    input_image = Image.open(input_path).convert("RGB")
                    physx_dir = vis.find_physx3d_sample(physx3d_root, spec.physx3d_dirs, stem)
                    anything_dir = anything_root / anything_name / stem

                    selected_frame = 0
                    view_iou = 0.0
                    rgb_frame = None
                    if physx_dir and (physx_dir / "rgb.mp4").is_file():
                        key = (physx_dir / "rgb.mp4", input_path)
                        if key not in view_cache:
                            view_cache[key] = vis.closest_render_frame(
                                key[0], input_image, temporary_root, f"{dataset_name}_{stem}"
                            )
                        selected_frame, view_iou = view_cache[key]
                        rgb_key = (physx_dir / "rgb.mp4", selected_frame)
                        rgb_path = temporary_root / f"{dataset_name}_{stem}_rgb_{selected_frame}.png"
                        if rgb_key not in extracted_cache:
                            extracted_cache[rgb_key] = vis.extract_video_frame(rgb_key[0], selected_frame, rgb_path)
                        rgb_frame = extracted_cache[rgb_key]

                    # PhysX-3D source-video maps: recover at original source resolution.
                    if physx_dir and rgb_frame is not None:
                        for property_name, video_name in (
                            ("affordance", "affordance.mp4"),
                            ("material", "material.mp4"),
                        ):
                            video = physx_dir / video_name
                            if not video.is_file():
                                continue
                            key = (video, selected_frame)
                            frame_path = temporary_root / f"{dataset_name}_{stem}_{property_name}_{selected_frame}.png"
                            if key not in extracted_cache:
                                extracted_cache[key] = vis.extract_video_frame(video, selected_frame, frame_path)
                            heatmap = physx3d_source_heatmap(extracted_cache[key], rgb_frame, args.size)
                            bar = source_numeric_colorbar(
                                extracted_cache[key], args.size, property_name.title()
                            )
                            directory = output_root / variant / dataset_name / stem / "physx3d" / property_name
                            heatmap_path, colorbar_path = save_pair(directory, heatmap, bar)
                            rows.append({
                                "variant": variant, "dataset": dataset_name, "sample": stem,
                                "method": "physx3d", "property": property_name,
                                "source": "original MP4 frame + same-frame RGB mask",
                                "frame": selected_frame, "input_iou": view_iou,
                                "heatmap": str(heatmap_path), "colorbar": str(colorbar_path),
                            })

                    # CoVeTwin maps directly rasterize high-resolution URDF visual meshes.
                    info_path = anything_dir / "basic_info.json"
                    urdf_path = anything_dir / "basic.urdf"
                    if info_path.is_file() and urdf_path.is_file():
                        metadata = json.loads(info_path.read_text(encoding="utf-8"))
                        parts = list(metadata.get("parts") or [])
                        target = vis.select_target(parts, None)
                        affordance, _, material = vis.heat_values(parts, target, "young")
                        override = orientation_cache.get(sample_key) if variant == "bad" else None
                        mesh_parts, _, orientation = vis.aligned_part_arrays(
                            anything_dir, physx_dir, metric, alignment_args, input_image, override
                        )
                        if variant == "good" or sample_key not in orientation_cache:
                            orientation_cache[sample_key] = orientation
                        twinx_frame = orientation[1]
                        ids = vis.render_part_id_buffer(
                            mesh_parts,
                            twinx_frame,
                            args.supersampling,
                            canvas_size=(args.size, args.size),
                            center_x_fraction=0.5,
                        )
                        for property_name, values, ticks in (
                            ("affordance", affordance, [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]),
                            ("material", material, [(0.0, "-3"), (0.5, "0"), (1.0, "3")]),
                        ):
                            heatmap = direct_twinx_heatmap(ids, values, args.size)
                            bar = clean_colorbar(args.size, ticks, property_name.title())
                            directory = output_root / variant / dataset_name / stem / "physx_anything" / property_name
                            heatmap_path, colorbar_path = save_pair(directory, heatmap, bar)
                            rows.append({
                                "variant": variant, "dataset": dataset_name, "sample": stem,
                                "method": "physx_anything", "property": property_name,
                                "source": "direct high-resolution basic.urdf visual-mesh rasterization",
                                "frame": twinx_frame, "input_iou": orientation[2],
                                "heatmap": str(heatmap_path), "colorbar": str(colorbar_path),
                            })

        if args.include_ablation or args.ablation_only:
            ablation_root = args.ablation_root.resolve()
            variant_roots = sorted(
                path for path in ablation_root.glob("test_*") if path.is_dir()
            )
            print(f"Ablation roots discovered: {len(variant_roots)}", flush=True)
            for variant_root in variant_roots:
                dataset_name = (
                    "demo_new" if variant_root.name.startswith("test_demo_new_") else "demo"
                )
                spec = vis.DATASETS[dataset_name]
                input_dir = physx3d_root / spec.input_dir
                reference_root = anything_root / spec.anything_good
                stems = sorted(
                    (
                        path.name
                        for path in variant_root.iterdir()
                        if path.is_dir() and not path.name.startswith(".")
                    ),
                    key=vis.natural_key,
                )
                if selected_ids:
                    stems = [stem for stem in stems if stem in selected_ids]
                for index, stem in enumerate(stems, start=1):
                    sample_key = f"{dataset_name}/{stem}"
                    print(
                        f"[ablation {variant_root.name} {index}/{len(stems)}] {stem}",
                        flush=True,
                    )
                    input_path = vis.find_input(input_dir, stem)
                    if input_path is None:
                        print(f"[WARN] missing input image for {sample_key}", flush=True)
                        continue
                    input_image = Image.open(input_path).convert("RGB")
                    sample_dir = variant_root / stem
                    info_path = sample_dir / "basic_info.json"
                    urdf_path = sample_dir / "basic.urdf"
                    if not info_path.is_file() or not urdf_path.is_file():
                        print(f"[WARN] incomplete ablation prediction: {sample_dir}", flush=True)
                        continue

                    # Establish the pose once from full CoVeTwin, then reuse it for
                    # every representation/no-verification variant.  This makes
                    # differences in the output images attributable to geometry,
                    # not a separately selected camera.
                    if sample_key not in orientation_cache:
                        reference_dir = reference_root / stem
                        orientation_source = (
                            reference_dir
                            if (reference_dir / "basic.urdf").is_file()
                            else sample_dir
                        )
                        _, _, reference_orientation = vis.aligned_part_arrays(
                            orientation_source,
                            None,
                            metric,
                            alignment_args,
                            input_image,
                            None,
                        )
                        orientation_cache[sample_key] = reference_orientation
                    orientation = orientation_cache[sample_key]

                    metadata = json.loads(info_path.read_text(encoding="utf-8"))
                    parts = list(metadata.get("parts") or [])
                    target = vis.select_target(parts, None)
                    affordance, _, material = vis.heat_values(parts, target, "young")
                    mesh_parts, _, _ = vis.aligned_part_arrays(
                        sample_dir,
                        None,
                        metric,
                        alignment_args,
                        input_image,
                        orientation,
                    )
                    ids = vis.render_part_id_buffer(
                        mesh_parts,
                        orientation[1],
                        args.supersampling,
                        canvas_size=(args.size, args.size),
                        center_x_fraction=0.5,
                    )
                    for property_name, values, ticks in (
                        ("affordance", affordance, [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]),
                        ("material", material, [(0.0, "-3"), (0.5, "0"), (1.0, "3")]),
                    ):
                        heatmap = direct_twinx_heatmap(ids, values, args.size)
                        bar = clean_colorbar(args.size, ticks, property_name.title())
                        directory = (
                            output_root
                            / "ablation"
                            / variant_root.name
                            / dataset_name
                            / stem
                            / "physx_anything"
                            / property_name
                        )
                        heatmap_path, colorbar_path = save_pair(directory, heatmap, bar)
                        rows.append({
                            "variant": variant_root.name,
                            "dataset": dataset_name,
                            "sample": stem,
                            "method": "physx_anything_ablation",
                            "property": property_name,
                            "source": "direct ablation basic.urdf mesh rasterization; full CoVeTwin pose reused",
                            "frame": orientation[1],
                            "input_iou": orientation[2],
                            "heatmap": str(heatmap_path),
                            "colorbar": str(colorbar_path),
                        })

    manifest = output_root / ("ablation_manifest.csv" if args.ablation_only else "manifest.csv")
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "dataset", "sample", "method", "property", "source", "frame", "input_iou", "heatmap", "colorbar"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated direct panel pairs: {len(rows)}")
    print(f"Output: {output_root}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
