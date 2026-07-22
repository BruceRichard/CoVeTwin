#!/usr/bin/env python3
"""Build aligned PhysX-3D vs PhysX-Anything mesh heatmap figures.

PhysX-3D panels are read from the existing MP4 files.  PhysX-Anything panels
are rendered from the fine visual meshes referenced by ``basic.urdf``; voxel
coordinates are deliberately not used. The CoVeTwin mesh is rigidly aligned to
the corresponding PhysX-3D mesh and viewed with the exact yaw/pitch/FOV used
for the selected frame of PhysX-3D's 30-frame turntable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_dilation, binary_fill_holes


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_NAMES = ("affordance.mp4", "material.mp4")
PANEL_SIZE = (340, 300)
INPUT_WIDTH = 260
HEADER_HEIGHT = 96
FOOTER_HEIGHT = 74
CANVAS_HEIGHT = HEADER_HEIGHT + PANEL_SIZE[1] + FOOTER_HEIGHT


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_dir: str
    physx3d_dirs: tuple[str, ...]
    anything_good: str
    anything_bad: str


DATASETS = {
    "demo": DatasetSpec(
        name="demo",
        input_dir="demo",
        physx3d_dirs=("outputs_demo", "outputs_demo_urdf"),
        anything_good="test_demo",
        anything_bad="test_demo_bad",
    ),
    "demo_new": DatasetSpec(
        name="demo_new",
        input_dir="demo_new",
        physx3d_dirs=("outputs_demo_new", "outputs_demo_new_urdf"),
        anything_good="test_demo_new",
        anything_bad="test_demo_new_bad",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PhysX-3D project root.",
    )
    parser.add_argument(
        "--anything-root",
        type=Path,
        default=Path("/mnt/data/zhangzhaodong/PhysX-Anything"),
        help="PhysX-Anything project root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qualitative_heatmaps"),
        help="Output directory, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("good", "bad"),
        default=["good", "bad"],
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help=(
            "Zero-based PhysX-3D turntable frame. -1 automatically chooses the "
            "rgb.mp4 frame whose normalized silhouette best matches the input render."
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        help='Optional JSON mapping such as {"demo/0": 1, "demo_new/100047": 0}.',
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional sample stems to render, for example --only 0 920.",
    )
    parser.add_argument(
        "--material-property",
        choices=("young", "density"),
        default="young",
        help="PhysX-Anything property shown in the material panel.",
    )
    parser.add_argument("--azimuth", type=float, default=25.0)
    parser.add_argument("--elevation", type=float, default=15.0)
    parser.add_argument(
        "--mesh-alignment",
        choices=("input_view", "physx3d_icp", "canonical"),
        default="input_view",
        help=(
            "input_view searches discrete mesh rotations/cameras against the input "
            "silhouette and avoids ambiguous 3D ICP flips."
        ),
    )
    parser.add_argument("--alignment-samples", type=int, default=4000)
    parser.add_argument("--icp-iterations", type=int, default=20)
    parser.add_argument("--icp-candidates", type=int, default=6)
    parser.add_argument(
        "--mesh-supersampling",
        type=int,
        default=2,
        help="Software-rasterization supersampling for clean mesh silhouettes.",
    )
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Skip samples without all three PhysX-3D MP4 files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_SMALL = load_font(14)
FONT_BODY = load_font(17)
FONT_LABEL = load_font(18, bold=True)
FONT_TITLE = load_font(23, bold=True)


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  font: ImageFont.ImageFont, fill: str = "black") -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def contain(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    result = Image.new("RGB", size, background)
    fitted = ImageOps.contain(image.convert("RGBA"), size, Image.Resampling.LANCZOS)
    position = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    result.paste(fitted.convert("RGB"), position, fitted.getchannel("A"))
    return result


def find_input(input_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def find_physx3d_sample(project_root: Path, directories: Iterable[str], stem: str) -> Path | None:
    for directory in directories:
        sample_dir = project_root / directory / stem
        if all((sample_dir / name).is_file() for name in VIDEO_NAMES):
            return sample_dir
    return None


def extract_video_frame(video: Path, frame: int, destination: Path) -> Image.Image:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not destination.exists():
        message = completed.stderr.strip() or "ffmpeg did not produce an image"
        raise RuntimeError(f"Cannot read frame {frame} from {video}: {message}")
    with Image.open(destination) as image:
        return image.convert("RGB")


def normalized_silhouette(image: Image.Image, size: int = 128) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    mask = distance > 24.0
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.zeros((size, size), dtype=bool)
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    cropped = Image.fromarray((mask[top:bottom, left:right] * 255).astype(np.uint8))
    contained = ImageOps.contain(cropped, (size, size), Image.Resampling.NEAREST)
    result = Image.new("L", (size, size), 0)
    result.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return np.asarray(result) > 127


def transparent_input_render(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    alpha = np.clip((distance - 8.0) * (255.0 / 28.0), 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack((rgb.astype(np.uint8), alpha))
    return Image.fromarray(rgba, "RGBA")


def transparent_physx3d_panel(heatmap: Image.Image, rgb_frame: Image.Image) -> Image.Image:
    """Remove the plot background while preserving genuinely black heat values."""
    heat = np.asarray(heatmap.convert("RGB"), dtype=np.uint8)
    near_black = np.max(heat, axis=2) < 18
    row_indices = np.where(near_black.mean(axis=1) > 0.25)[0]
    column_indices = np.where(near_black.mean(axis=0) > 0.25)[0]
    if not len(row_indices) or not len(column_indices):
        return heatmap.convert("RGBA")
    top, bottom = int(row_indices.min()), int(row_indices.max()) + 1
    left, right = int(column_indices.min()), int(column_indices.max()) + 1

    source = np.asarray(rgb_frame.convert("RGB"), dtype=np.uint8)
    foreground = (np.max(source, axis=2) > 8).astype(np.uint8) * 255
    mask = Image.fromarray(foreground, "L").resize(
        (right - left, bottom - top), Image.Resampling.LANCZOS
    )
    alpha = np.zeros(heat.shape[:2], dtype=np.uint8)
    # Preserve annotations and colorbars, but make white page margins transparent.
    nonwhite = np.min(heat, axis=2) < 245
    alpha[nonwhite] = 255
    alpha[top:bottom, left:right] = np.asarray(mask)
    return Image.fromarray(np.dstack((heat, alpha)), "RGBA")


def silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second)
    if not np.any(union):
        return 0.0
    return float(np.logical_and(first, second).sum() / union.sum())


def closest_render_frame(
    rgb_video: Path, input_image: Image.Image, temporary_root: Path, cache_key: str
) -> tuple[int, float]:
    frame_dir = temporary_root / f"viewmatch_{cache_key}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "%03d.png"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(rgb_video),
        "-frames:v", "30", str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    paths = sorted(frame_dir.glob("*.png"))
    if completed.returncode != 0 or not paths:
        message = completed.stderr.strip() or "ffmpeg produced no RGB frames"
        print(f"[WARN] automatic view matching failed for {rgb_video}: {message}", flush=True)
        return 0, 0.0
    target = normalized_silhouette(input_image)
    scores = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            scores.append((silhouette_iou(target, normalized_silhouette(image)), index))
    score, index = max(scores)
    return index, score


def placeholder(text: str, size: tuple[int, int] = PANEL_SIZE) -> Image.Image:
    image = Image.new("RGB", size, "#f2f2f2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#b8b8b8", width=2)
    centered_text(draw, (24, 20, size[0] - 24, size[1] - 20), text, FONT_BODY, "#555555")
    return image


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else None


def part_young_modulus(part: dict[str, object]) -> float | None:
    for key, value in part.items():
        if str(key).lower().startswith("young"):
            return parse_number(value)
    return None


def select_target(parts: list[dict[str, object]], requested_label: int | None) -> int:
    labels = [int(part.get("label", index)) for index, part in enumerate(parts)]
    if requested_label in labels:
        return int(requested_label)
    ranked = sorted(
        zip(parts, labels),
        key=lambda item: (parse_number(item[0].get("priority_rank")) or math.inf, item[1]),
    )
    return ranked[0][1] if ranked else 0


def heat_values(parts: list[dict[str, object]], target_label: int,
                material_property: str) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    ranks = [parse_number(part.get("priority_rank")) for part in parts]
    finite_ranks = [rank for rank in ranks if rank is not None]
    maximum_rank = max(finite_ranks, default=1.0)
    affordance: dict[int, float] = {}
    description: dict[int, float] = {}
    material: dict[int, float] = {}
    for index, part in enumerate(parts):
        label = int(part.get("label", index))
        rank = parse_number(part.get("priority_rank")) or maximum_rank
        affordance[label] = 1.0 if maximum_rank <= 1 else 1.0 - (rank - 1.0) / (maximum_rank - 1.0)
        description[label] = 1.0 if label == target_label else 0.0
        if material_property == "density":
            density = parse_number(part.get("density"))
            material[label] = min(max((density or 0.0) / 20.0, 0.0), 1.0)
        else:
            young = part_young_modulus(part)
            log_young = math.log10(max(young or 1e-3, 1e-3))
            material[label] = min(max((log_young + 3.0) / 6.0, 0.0), 1.0)
    return affordance, description, material


def visual_label(visual: object, fallback: int) -> int:
    """Recover the VLM part label, independent of URDF link ordering."""
    stem = Path(str(getattr(visual, "path"))).stem
    if stem.isdigit():
        return int(stem)
    match = re.fullmatch(r"l_(\d+)", str(getattr(visual, "link_name", "")))
    return int(match.group(1)) if match else fallback


def load_urdf_mesh_parts(sample_dir: Path, metric_module: object) -> dict[int, object]:
    urdf = sample_dir / "basic.urdf"
    if not urdf.is_file():
        return {}
    asset = metric_module.parse_urdf_asset(urdf.resolve(), "twinx_heatmap")
    grouped: dict[int, list[object]] = {}
    for index, visual in enumerate(asset.render_visuals):
        label = visual_label(visual, index)
        try:
            mesh = metric_module.transformed_mesh(visual.path, visual.transform)
        except Exception as exc:
            print(f"[WARN] cannot load URDF visual {visual.path}: {exc}", flush=True)
            continue
        grouped.setdefault(label, []).append(mesh)
    return {
        label: metric_module.trimesh.util.concatenate(meshes)
        for label, meshes in grouped.items()
        if meshes
    }


def physx3d_reference_mesh(sample_dir: Path | None, metric_module: object) -> object | None:
    if sample_dir is None:
        return None
    for path in (sample_dir / "texture.glb", sample_dir / "kinematic.obj"):
        if path.is_file():
            try:
                return metric_module.load_mesh_files([path])
            except Exception as exc:
                print(f"[WARN] cannot load PhysX-3D reference {path}: {exc}", flush=True)
    group_paths = sorted((sample_dir / "urdf_export" / "objs").glob("*.obj"))
    if group_paths:
        try:
            return metric_module.load_mesh_files(group_paths)
        except Exception as exc:
            print(f"[WARN] cannot load PhysX-3D URDF meshes: {exc}", flush=True)
    return None


def normalized_projected_points(
    points: np.ndarray, frame: int, size: int = 128
) -> np.ndarray:
    origin, right, up = physx3d_camera(frame)
    forward = -origin / np.linalg.norm(origin)
    relative = points - origin
    depth = relative @ forward
    valid = depth > 1e-5
    relative = relative[valid]
    depth = depth[valid]
    focal = 0.5 * size / math.tan(math.radians(40.0) / 2.0)
    horizontal = np.rint(size * 0.5 + focal * (relative @ right) / depth).astype(int)
    vertical = np.rint(size * 0.5 - focal * (relative @ up) / depth).astype(int)
    inside = (
        (horizontal >= 0)
        & (horizontal < size)
        & (vertical >= 0)
        & (vertical < size)
    )
    mask = np.zeros((size, size), dtype=bool)
    mask[vertical[inside], horizontal[inside]] = True
    mask = binary_fill_holes(binary_dilation(mask, iterations=2))
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return mask
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    cropped = Image.fromarray(
        (mask[top:bottom, left:right] * 255).astype(np.uint8), "L"
    )
    fitted = ImageOps.contain(cropped, (size, size), Image.Resampling.NEAREST)
    result = Image.new("L", (size, size), 0)
    result.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return np.asarray(result) > 127


def select_twinx_input_orientation(
    normalized_points: np.ndarray,
    input_image: Image.Image,
    metric_module: object,
) -> tuple[np.ndarray, int, float]:
    target = normalized_silhouette(input_image)
    best_score = -1.0
    best_rotation = np.eye(3, dtype=np.float64)
    best_frame = 0
    for rotation in metric_module.cube_rotations():
        rotated = normalized_points @ rotation.T
        for frame in range(30):
            silhouette = normalized_projected_points(rotated, frame)
            score = silhouette_iou(target, silhouette)
            if score > best_score:
                best_score = score
                best_rotation = np.asarray(rotation, dtype=np.float64)
                best_frame = frame
    return best_rotation, best_frame, best_score


def aligned_part_arrays(
    sample_dir: Path,
    physx3d_dir: Path | None,
    metric_module: object,
    args: argparse.Namespace,
    input_image: Image.Image,
    orientation_override: tuple[np.ndarray, int, float] | None = None,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    str,
    tuple[np.ndarray, int, float],
]:
    meshes = load_urdf_mesh_parts(sample_dir, metric_module)
    if not meshes:
        return {}, "URDF visual meshes unavailable"
    combined = metric_module.trimesh.util.concatenate(list(meshes.values()))
    normalization = metric_module.mesh_normalization(combined)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    alignment_note = "canonical normalized URDF frame"
    twinx_frame = 0
    view_iou = 0.0

    reference = physx3d_reference_mesh(physx3d_dir, metric_module)
    if args.mesh_alignment == "input_view":
        if orientation_override is None:
            normalized_points = normalization.points(
                metric_module.sample_surface(
                    combined, max(6000, args.alignment_samples), 2026
                )
            )
            rotation, twinx_frame, view_iou = select_twinx_input_orientation(
                normalized_points, input_image, metric_module
            )
        else:
            rotation, twinx_frame, view_iou = orientation_override
        alignment_note = (
            f"CoVeTwin f{twinx_frame}, input IoU={view_iou:.3f} "
            f"(image-matched; no free ICP flip)"
        )
    elif args.mesh_alignment == "physx3d_icp" and reference is not None:
        reference_normalization = metric_module.mesh_normalization(reference)
        source_points = normalization.points(
            metric_module.sample_surface(combined, args.alignment_samples, 2026)
        )
        target_points = reference_normalization.points(
            metric_module.sample_surface(reference, args.alignment_samples, 2027)
        )
        alignment = metric_module.estimate_alignment(
            source_points,
            target_points,
            argparse.Namespace(
                alignment="cube_icp",
                alignment_samples=args.alignment_samples,
                icp_iterations=args.icp_iterations,
                icp_candidates=args.icp_candidates,
            ),
            seed=2028,
        )
        rotation = alignment.rotation
        translation = alignment.translation
        alignment_note = f"legacy PhysX-3D mesh ICP (score={alignment.score:.5f})"

    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for label, mesh in meshes.items():
        vertices = normalization.points(np.asarray(mesh.vertices, dtype=np.float64))
        vertices = vertices @ rotation.T + translation
        arrays[label] = (vertices, np.asarray(mesh.faces, dtype=np.int64))
    orientation = (np.asarray(rotation), int(twinx_frame), float(view_iou))
    return arrays, alignment_note, orientation


def physx3d_camera(frame: int, number_of_frames: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 2.0 * 3.1415, number_of_frames)[frame % number_of_frames]
    pitch = 0.25 + 0.5 * math.sin(float(phase))
    origin = np.asarray(
        [math.sin(phase) * math.cos(pitch), math.cos(phase) * math.cos(pitch), math.sin(pitch)],
        dtype=np.float64,
    ) * 2.0
    forward = -origin / np.linalg.norm(origin)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return origin, right, up


def render_part_id_buffer(
    parts: dict[int, tuple[np.ndarray, np.ndarray]],
    frame: int,
    supersampling: int,
    canvas_size: tuple[int, int] = PANEL_SIZE,
    center_x_fraction: float = 0.42,
) -> np.ndarray:
    factor = max(1, int(supersampling))
    width, height = canvas_size[0] * factor, canvas_size[1] * factor
    origin, right, up = physx3d_camera(frame)
    forward = -origin / np.linalg.norm(origin)
    focal = 0.5 * width / math.tan(math.radians(40.0) / 2.0)
    polygons: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for label, (vertices, faces) in parts.items():
        relative = vertices - origin
        z = relative @ forward
        valid_vertices = z > 1e-5
        x = relative @ right
        y = relative @ up
        projected = np.column_stack(
            (
                width * center_x_fraction + focal * x / np.maximum(z, 1e-5),
                height * 0.50 - focal * y / np.maximum(z, 1e-5),
            )
        )
        valid_faces = np.all(valid_vertices[faces], axis=1)
        usable = faces[valid_faces]
        if not len(usable):
            continue
        polygons.append(projected[usable])
        depths.append(z[usable].mean(axis=1))
        labels.append(np.full(len(usable), int(label), dtype=np.int64))

    if not polygons:
        return np.zeros((height, width), dtype=np.int32)
    all_polygons = np.concatenate(polygons)
    all_depths = np.concatenate(depths)
    all_labels = np.concatenate(labels)
    order = np.argsort(all_depths)[::-1]  # painter: far to near
    ids = all_labels[order] + 1
    encoded = np.column_stack(
        ((ids & 255), ((ids >> 8) & 255), ((ids >> 16) & 255))
    ).astype(np.float64) / 255.0

    figure = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    figure.patch.set_facecolor("black")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_facecolor("black")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    axis.add_collection(
        PolyCollection(
            all_polygons[order], facecolors=encoded, edgecolors="none", antialiaseds=False
        )
    )
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    return (
        rgb[..., 0].astype(np.int32)
        + (rgb[..., 1].astype(np.int32) << 8)
        + (rgb[..., 2].astype(np.int32) << 16)
    )


def draw_colorbar(draw: ImageDraw.ImageDraw, normalized_ticks: list[tuple[float, str]]) -> None:
    cmap = colormaps["jet"]
    left, top, right, bottom = 294, 30, 310, 260
    for y in range(top, bottom):
        value = 1.0 - (y - top) / max(bottom - top - 1, 1)
        color = tuple(int(channel * 255) for channel in cmap(value)[:3])
        draw.line((left, y, right, y), fill=color)
    draw.rectangle((left, top, right, bottom), outline="black", width=1)
    for value, label in normalized_ticks:
        y = int(bottom - value * (bottom - top))
        draw.line((right + 1, y, right + 6, y), fill="black", width=1)
        draw.text((right + 8, y - 7), label, font=FONT_SMALL, fill="black")


def render_mesh_heatmap(
    part_ids: np.ndarray,
    values: dict[int, float],
    ticks: list[tuple[float, str]],
    supersampling: int,
) -> Image.Image:
    if part_ids.size == 0 or not np.any(part_ids):
        return placeholder("PhysX-Anything URDF mesh unavailable")
    rgb = np.full((*part_ids.shape, 3), 255, dtype=np.uint8)
    cmap = colormaps["jet"]
    for label in np.unique(part_ids):
        if label <= 0:
            continue
        original_label = int(label) - 1
        value = float(np.clip(values.get(original_label, 0.0), 0.0, 1.0))
        color = (0, 0, 0) if value <= 0.0 else tuple(int(channel * 255) for channel in cmap(value)[:3])
        rgb[part_ids == label] = color
    image = Image.fromarray(rgb, "RGB")
    if max(1, int(supersampling)) > 1:
        image = image.resize(PANEL_SIZE, Image.Resampling.LANCZOS)
    elif image.size != PANEL_SIZE:
        image = image.resize(PANEL_SIZE, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw_colorbar(draw, ticks)
    return image


def target_metadata(parts: list[dict[str, object]], target_label: int) -> tuple[str, str]:
    for index, part in enumerate(parts):
        if int(part.get("label", index)) == target_label:
            name = str(part.get("name") or f"part {target_label}")
            description = str(part.get("Basic_description") or name)
            return name, description
    return f"part {target_label}", f"part {target_label}"


def compose_figure(
    input_image: Image.Image,
    physx3d_panels: list[Image.Image],
    anything_panels: list[Image.Image],
    metadata: dict[str, object],
    sample_key: str,
    variant: str,
    material_property: str,
    alignment_note: str,
) -> Image.Image:
    gap = 10
    width = INPUT_WIDTH + 4 * PANEL_SIZE[0] + 4 * gap
    canvas = Image.new("RGB", (width, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    x_physx3d = INPUT_WIDTH + gap
    x_anything = x_physx3d + 2 * (PANEL_SIZE[0] + gap)

    centered_text(draw, (x_physx3d, 4, x_anything - gap, 38), "PhysX-3D", FONT_TITLE)
    method = "PhysX-Anything" if variant == "good" else "PhysX-Anything (bad)"
    centered_text(draw, (x_anything, 4, width, 38), method, FONT_TITLE)
    object_name = str(metadata.get("object_name") or "Unknown object")
    dimension = str(metadata.get("dimension") or "unknown")
    subtitle = f"{sample_key}  |  {object_name}  |  Dimension: {dimension}"
    centered_text(draw, (0, 40, width, 72), subtitle, FONT_BODY, "#174a8b")

    panel_y = HEADER_HEIGHT
    canvas.paste(contain(input_image, (INPUT_WIDTH, PANEL_SIZE[1]), "white"), (0, panel_y))
    centered_text(draw, (0, panel_y + PANEL_SIZE[1], INPUT_WIDTH, CANVAS_HEIGHT), "Input", FONT_LABEL)

    labels = ("Affordance", "Material")
    for group_index, panels in enumerate((physx3d_panels, anything_panels)):
        group_x = x_physx3d if group_index == 0 else x_anything
        for index, panel in enumerate(panels):
            x = group_x + index * (PANEL_SIZE[0] + gap)
            canvas.paste(contain(panel, PANEL_SIZE), (x, panel_y))
            centered_text(
                draw,
                (x, panel_y + PANEL_SIZE[1], x + PANEL_SIZE[0], panel_y + PANEL_SIZE[1] + 31),
                labels[index],
                FONT_LABEL,
                ("#d94801" if group_index == 0 else "#008f39"),
            )

    property_label = "log10 Young's modulus (GPa)" if material_property == "young" else "density (g/cm3)"
    note = f"Material: {property_label}  |  View/alignment: {alignment_note}"
    centered_text(draw, (INPUT_WIDTH, CANVAS_HEIGHT - 34, width, CANVAS_HEIGHT), note, FONT_SMALL, "#444444")
    return canvas


def make_contact_sheet(images: list[Path], destination: Path) -> None:
    if not images:
        return
    columns = 2
    thumb_width = 1100
    thumbnails: list[Image.Image] = []
    for path in images:
        with Image.open(path) as image:
            ratio = thumb_width / image.width
            thumbnails.append(image.convert("RGB").resize((thumb_width, int(image.height * ratio)), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbnails)
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
    for index, image in enumerate(thumbnails):
        x = index % columns * thumb_width
        y = index // columns * cell_height
        sheet.paste(image, (x, y))
    sheet.save(destination, quality=92)


def load_targets(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    content = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in content.items()}


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    anything_root = args.anything_root.resolve()
    if str(anything_root) not in sys.path:
        sys.path.insert(0, str(anything_root))
    try:
        from evaluation import evaluate_metrics as metric_module
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import {anything_root / 'evaluation/evaluate_metrics.py'}: {exc}"
        ) from exc
    output_root = args.output if args.output.is_absolute() else project_root / args.output
    output_root.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets)
    rows: list[dict[str, object]] = []
    extracted_cache: dict[tuple[Path, int], Image.Image] = {}
    view_match_cache: dict[tuple[Path, Path | None], tuple[int, float]] = {}
    twinx_orientation_cache: dict[str, tuple[np.ndarray, int, float]] = {}

    with tempfile.TemporaryDirectory(prefix="physx_heatmaps_") as temporary:
        temporary_root = Path(temporary)
        for dataset_name in args.datasets:
            spec = DATASETS[dataset_name]
            input_dir = project_root / spec.input_dir
            stems = sorted(
                {path.stem for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS},
                key=natural_key,
            )
            if args.only:
                selected = set(args.only)
                stems = [stem for stem in stems if stem in selected]
            for variant in args.variants:
                anything_dir_name = spec.anything_good if variant == "good" else spec.anything_bad
                variant_output = output_root / variant / dataset_name
                variant_output.mkdir(parents=True, exist_ok=True)
                generated: list[Path] = []
                for sample_index, stem in enumerate(stems, start=1):
                    sample_key = f"{dataset_name}/{stem}"
                    destination = variant_output / f"{stem}.png"
                    input_path = find_input(input_dir, stem)
                    physx3d_dir = find_physx3d_sample(project_root, spec.physx3d_dirs, stem)
                    anything_dir = anything_root / anything_dir_name / stem
                    anything_complete = all(
                        (anything_dir / name).is_file()
                        for name in ("basic_info.json", "basic.urdf")
                    )
                    status = "complete" if physx3d_dir and anything_complete else "partial"
                    if args.only_complete and status != "complete":
                        rows.append({
                            "dataset": dataset_name, "variant": variant, "sample": stem,
                            "status": "skipped", "physx3d": bool(physx3d_dir),
                            "physx_anything": anything_complete, "output": "",
                        })
                        continue
                    if destination.exists() and not args.overwrite:
                        generated.append(destination)
                        rows.append({
                            "dataset": dataset_name, "variant": variant, "sample": stem,
                            "status": "existing", "physx3d": bool(physx3d_dir),
                            "physx_anything": anything_complete, "output": str(destination),
                        })
                        continue

                    print(f"[{variant} {dataset_name} {sample_index}/{len(stems)}] {stem}", flush=True)
                    input_image = Image.open(input_path).convert("RGB") if input_path else placeholder("Input unavailable")
                    selected_frame = max(0, args.frame)
                    view_score = None
                    if args.frame < 0 and physx3d_dir and (physx3d_dir / "rgb.mp4").is_file():
                        match_key = (physx3d_dir / "rgb.mp4", input_path)
                        if match_key not in view_match_cache:
                            view_match_cache[match_key] = closest_render_frame(
                                physx3d_dir / "rgb.mp4",
                                input_image,
                                temporary_root,
                                f"{dataset_name}_{stem}",
                            )
                        selected_frame, view_score = view_match_cache[match_key]
                        print(
                            f"  closest input-render view: frame={selected_frame}, "
                            f"silhouette_IoU={view_score:.4f}",
                            flush=True,
                        )
                    physx3d_panels: list[Image.Image] = []
                    if physx3d_dir:
                        rgb_frame = None
                        rgb_video = physx3d_dir / "rgb.mp4"
                        if rgb_video.is_file():
                            rgb_key = (rgb_video, selected_frame)
                            if rgb_key not in extracted_cache:
                                rgb_path = temporary_root / f"{dataset_name}_{stem}_rgb_{selected_frame}.png"
                                extracted_cache[rgb_key] = extract_video_frame(
                                    rgb_video, selected_frame, rgb_path
                                )
                            rgb_frame = extracted_cache[rgb_key]
                        for video_name in VIDEO_NAMES:
                            video = physx3d_dir / video_name
                            cache_key = (video, selected_frame)
                            if cache_key not in extracted_cache:
                                frame_path = temporary_root / f"{dataset_name}_{stem}_{video.stem}_{selected_frame}.png"
                                extracted_cache[cache_key] = extract_video_frame(video, selected_frame, frame_path)
                            panel = extracted_cache[cache_key]
                            if rgb_frame is not None:
                                panel = transparent_physx3d_panel(panel, rgb_frame)
                            physx3d_panels.append(panel)
                    else:
                        physx3d_panels = [placeholder("PhysX-3D video unavailable") for _ in VIDEO_NAMES]

                    metadata: dict[str, object] = {}
                    alignment_note = "unavailable"
                    if anything_complete:
                        metadata = json.loads((anything_dir / "basic_info.json").read_text(encoding="utf-8"))
                        parts = list(metadata.get("parts") or [])
                        target_label = select_target(parts, targets.get(sample_key))
                        orientation_override = (
                            twinx_orientation_cache.get(sample_key)
                            if variant == "bad"
                            else None
                        )
                        mesh_parts, alignment_note, orientation = aligned_part_arrays(
                            anything_dir,
                            physx3d_dir,
                            metric_module,
                            args,
                            input_image,
                            orientation_override,
                        )
                        if variant == "good" or sample_key not in twinx_orientation_cache:
                            twinx_orientation_cache[sample_key] = orientation
                        twinx_frame = orientation[1]
                        part_ids = render_part_id_buffer(
                            mesh_parts, twinx_frame, args.mesh_supersampling
                        )
                        affordance, _, material = heat_values(parts, target_label, args.material_property)
                        material_ticks = (
                            [(0.0, "-3"), (0.5, "0"), (1.0, "3")]
                            if args.material_property == "young"
                            else [(0.0, "0"), (0.5, "10"), (1.0, "20")]
                        )
                        anything_panels = [
                            render_mesh_heatmap(part_ids, affordance, [(0.0, "0"), (0.5, "0.5"), (1.0, "1")], args.mesh_supersampling),
                            render_mesh_heatmap(part_ids, material, material_ticks, args.mesh_supersampling),
                        ]
                        view_text = f"PhysX-3D f{selected_frame}"
                        if view_score is not None:
                            view_text += f", input IoU={view_score:.3f}"
                        alignment_note = f"{view_text}; {alignment_note}"
                    else:
                        anything_panels = [placeholder("PhysX-Anything data unavailable") for _ in VIDEO_NAMES]

                    figure = compose_figure(
                        transparent_input_render(input_image),
                        physx3d_panels,
                        anything_panels,
                        metadata,
                        sample_key, variant, args.material_property,
                        alignment_note,
                    )
                    figure.save(destination)
                    generated.append(destination)
                    rows.append({
                        "dataset": dataset_name, "variant": variant, "sample": stem,
                        "status": status, "physx3d": bool(physx3d_dir),
                        "physx_anything": anything_complete, "output": str(destination),
                    })
                make_contact_sheet(generated, output_root / f"index_{variant}_{dataset_name}.jpg")

    report_path = output_root / "manifest.csv"
    with report_path.open("w", encoding="utf-8", newline="") as report:
        writer = csv.DictWriter(
            report,
            fieldnames=("dataset", "variant", "sample", "status", "physx3d", "physx_anything", "output"),
        )
        writer.writeheader()
        writer.writerows(rows)

    completed = sum(row["status"] in {"complete", "existing"} and row["physx3d"] and row["physx_anything"] for row in rows)
    partial = sum(row["status"] == "partial" for row in rows)
    print(f"Generated/available complete comparisons: {completed}")
    print(f"Partial comparisons: {partial}")
    print(f"Output: {output_root}")
    print(f"Manifest: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
