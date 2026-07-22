#!/usr/bin/env python3
"""Render a compact side-by-side comparison from generated URDF assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True, metavar="LABEL=OBJECT_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="CoVeTwin post-hoc ablation comparison")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--yaw", type=float, default=-45.0)
    parser.add_argument("--pitch", type=float, default=-25.0)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def object_aabb(body: int) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for link in [-1, *range(p.getNumJoints(body))]:
        lower, upper = p.getAABB(body, link)
        mins.append(lower)
        maxs.append(upper)
    return np.min(mins, axis=0), np.max(maxs, axis=0)


def render_urdf(
    object_dir: Path,
    resolution: int,
    yaw: float,
    pitch: float,
) -> Image.Image:
    urdf = (object_dir / "basic.urdf").resolve()
    if not urdf.exists():
        raise FileNotFoundError(urdf)

    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True)
        lower, upper = object_aabb(body)
        center = (lower + upper) / 2.0
        extent = np.maximum(upper - lower, 1e-6)
        distance = max(0.5, float(np.max(extent)) * 2.5)
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=center.tolist(),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=1.0,
            nearVal=max(0.001, distance / 100.0),
            farVal=max(10.0, distance * 10.0),
        )
        _, _, rgba, _, segmentation = p.getCameraImage(
            width=resolution,
            height=resolution,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=[-2, -3, 5],
            shadow=1,
        )
        array = np.asarray(rgba, dtype=np.uint8).reshape(resolution, resolution, 4)
        mask = np.asarray(segmentation).reshape(resolution, resolution) < 0
        array[mask, :3] = 255
        array[..., 3] = 255
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    finally:
        p.disconnect(client)


def main() -> None:
    args = parse_args()
    parsed: list[tuple[str, Path]] = []
    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=OBJECT_DIR, got {item!r}")
        label, directory = item.split("=", 1)
        parsed.append((label, Path(directory)))

    panel_size = args.resolution
    title_height = 58
    label_height = 46
    gap = 12
    width = len(parsed) * panel_size + (len(parsed) + 1) * gap
    height = title_height + panel_size + label_height + 2 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    label_font = load_font(22)
    draw.text((gap, 14), args.title, fill=(25, 25, 25), font=title_font)

    for index, (label, directory) in enumerate(parsed):
        panel = render_urdf(directory, panel_size, args.yaw, args.pitch)
        x = gap + index * (panel_size + gap)
        y = title_height + gap
        canvas.paste(panel, (x, y))
        color = (29, 145, 65) if label.lower().startswith("full") else (205, 85, 50)
        draw.rectangle(
            (x, y, x + panel_size - 1, y + panel_size - 1),
            outline=color,
            width=5,
        )
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x + (panel_size - text_width) / 2, y + panel_size + 10),
            label,
            fill=color,
            font=label_font,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
