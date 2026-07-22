import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.voxel import ops as voxel_ops

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trellis.pipelines import TrellisImageTo3DPipeline


FLOW_MODEL_NAMES = (
    "sparse_structure_decoder",
    "sparse_structure_encoder",
    "sparse_structure_flow_model",
    "image_cond_model",
)


def normalized_centers(coords: np.ndarray, grid_size: int) -> np.ndarray:
    return (coords.astype(np.float32) + 0.5) / float(grid_size) - 0.5


def export_voxels(coords: np.ndarray, grid_size: int, output_prefix: str, color) -> None:
    coords = np.unique(np.asarray(coords, dtype=np.int64), axis=0)
    centers = normalized_centers(coords, grid_size)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(centers), 1))

    trimesh.points.PointCloud(centers, colors=colors).export(output_prefix + "_points.ply")
    voxel_mesh = voxel_ops.multibox(
        centers,
        pitch=1.0 / float(grid_size),
        colors=colors,
    )
    voxel_mesh.export(output_prefix + "_mesh.obj")


def export_overlay(coarse_64: np.ndarray, optimized_64: np.ndarray, output_path: str) -> None:
    coarse_centers = normalized_centers(coarse_64, 64)
    optimized_centers = normalized_centers(optimized_64, 64)
    points = np.concatenate([coarse_centers, optimized_centers], axis=0)
    colors = np.concatenate(
        [
            np.tile(np.array([55, 125, 255, 255], dtype=np.uint8), (len(coarse_centers), 1)),
            np.tile(np.array([255, 105, 45, 255], dtype=np.uint8), (len(optimized_centers), 1)),
        ],
        axis=0,
    )
    trimesh.points.PointCloud(points, colors=colors).export(output_path)


def sample_flow_voxels(
    pipeline_path: str,
    image_path: str,
    coarse_32: np.ndarray,
    seed: int,
    device: str,
) -> np.ndarray:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(pipeline_path)
    torch_device = torch.device(device)
    for name in FLOW_MODEL_NAMES:
        pipeline.models[name].to(torch_device)

    coarse_64 = coarse_32 + 16
    pointinput = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=torch_device)
    pointinput[:, :, coarse_64[:, 0], coarse_64[:, 1], coarse_64[:, 2]] = 1

    with torch.inference_mode():
        image = pipeline.preprocess_image(Image.open(image_path))
        cond = pipeline.get_cond([image])
        torch.manual_seed(seed)
        low = pipeline.models["sparse_structure_encoder"](
            pointinput,
            sample_posterior=False,
        )
        coords = pipeline.sample_sparse_structure_control(low, cond, 1, {})

    return coords[:, 1:].detach().cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the VLM coarse voxels and sparse-flow-refined voxels."
    )
    parser.add_argument("--sample_dir", default="test_demo/1")
    parser.add_argument("--image", default="demo/1.png")
    parser.add_argument("--pipeline", default="pretrain/decoder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sample_dir = os.path.abspath(args.sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    coarse_path = os.path.join(sample_dir, "allind.npy")
    coarse_32 = np.unique(np.load(coarse_path).astype(np.int64), axis=0)
    coarse_64 = coarse_32 + 16

    export_voxels(
        coarse_32,
        32,
        os.path.join(sample_dir, "coarse_voxels_32"),
        [55, 125, 255, 255],
    )
    np.save(os.path.join(sample_dir, "coarse_input_voxels_64.npy"), coarse_64)
    export_voxels(
        coarse_64,
        64,
        os.path.join(sample_dir, "coarse_input_voxels_64"),
        [55, 125, 255, 255],
    )

    optimized_64 = sample_flow_voxels(
        args.pipeline,
        args.image,
        coarse_32,
        args.seed,
        args.device,
    )
    optimized_64 = np.unique(optimized_64, axis=0)
    np.save(os.path.join(sample_dir, "flow_optimized_voxels_64.npy"), optimized_64)
    export_voxels(
        optimized_64,
        64,
        os.path.join(sample_dir, "flow_optimized_voxels_64"),
        [255, 105, 45, 255],
    )
    export_overlay(
        coarse_64,
        optimized_64,
        os.path.join(sample_dir, "coarse_vs_flow_points.ply"),
    )

    stats = {
        "seed": args.seed,
        "coarse_voxels_32": int(len(coarse_32)),
        "coarse_input_voxels_64": int(len(coarse_64)),
        "flow_optimized_voxels_64": int(len(optimized_64)),
        "coarse_32_bounds": [coarse_32.min(axis=0).tolist(), coarse_32.max(axis=0).tolist()],
        "flow_64_bounds": [optimized_64.min(axis=0).tolist(), optimized_64.max(axis=0).tolist()],
    }
    with open(os.path.join(sample_dir, "flow_voxel_stats.json"), "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
