#!/usr/bin/env python3
"""Evaluate CoVeTwin and articulated-asset baselines against PhysX-Mobility GT.

The adapters cover CoVeTwin-compatible output trees, URDF-Anything,
Articulate-Anything and PhysX-3D.
They convert method-specific meshes and URDFs to a shared zero-pose geometry,
part, articulation, rendering and executability protocol.

The geometry protocol is scale invariant: both surfaces are centered and
normalized by their largest bounding-box extent, then the prediction is
rigidly aligned to GT with cube-rotation initialization and trimmed ICP.
Scale is evaluated uniformly from raw mesh-coordinate extents calibrated by
the annotated GT largest dimension.  A method-reported dimension, when
present, is retained separately as ``reported_scale``.

PSNR uses the GT cameras in ``dataset_toolkits/renders_all``. Missing
prediction views are rendered by the sibling ``render_pred_views.py`` worker with
Blender and cached in the output directory.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import trimesh
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - gives a useful CLI failure
    raise SystemExit(
        "Missing evaluation dependency. Activate the CoVeTwin conda "
        "environment and install trimesh, Pillow and scipy. Original error: "
        f"{exc}"
    ) from exc


DEFAULT_TEST_DEMO_MAP = {
    "0": "1817",
    "1": "1986",
    "2": "2230",
    "3": "4533",
    "4": "11586",
    "5": "12654",
    "6": "102434",
    "7": "100197",
    "8": "100321",
    "9": "100443",
    "10": "100501",
    "11": "100523",
    "12": "100758",
    "13": "100907",
    "14": "100925",
    "15": "101448",
    "16": "101861",
}

GROUP_TYPE_NAMES = {
    "A": "floating",
    "B": "prismatic",
    "C": "revolute",
    "D": "ball",
    "E": "fixed",
    "CB": "revolute_prismatic",
}

SUMMARY_METRICS = [
    "geometry.psnr_db",
    "geometry.psnr_foreground_db",
    "geometry.chamfer_l2",
    "geometry.chamfer_l2_x1e3",
    "geometry.chamfer_l1",
    "geometry.fscore",
    "geometry.fscore_precision",
    "geometry.fscore_recall",
    "scale.absolute_scale_error_cm",
    "scale.max_dimension_absolute_error_cm",
    "scale.relative_scale_error",
    "semantics.material_macro_f1",
    "semantics.material_micro_f1",
    "semantics.affordance_macro_f1",
    "semantics.affordance_micro_f1",
    "semantics.part_match_coverage",
    "articulation.joint_type_accuracy",
    "articulation.axis_error_rad",
    "articulation.axis_error_deg",
    "articulation.origin_error_m",
    "articulation.motion_range_error",
    "articulation.revolute_motion_range_error_rad",
    "articulation.prismatic_motion_range_error_m",
    "executability.physics_engine_executable",
]


@dataclass
class Normalization:
    center: np.ndarray
    scale: float

    def points(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.center) / self.scale


@dataclass
class Alignment:
    rotation: np.ndarray
    translation: np.ndarray
    score: float

    def points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.rotation.T + self.translation

    def vector(self, vector: Sequence[float]) -> np.ndarray:
        return self.rotation @ normalize_vector(vector)


@dataclass
class Group:
    group_id: str
    members: set[int]
    parent: str | None
    params: list[float]
    code: str


@dataclass
class SampleSpec:
    adapter: str
    prediction_set: str
    root: Path
    sample_dir: Path
    gt_id: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    bundled_blender = (
        PROJECT_ROOT
        / "dataset_toolkits"
        / "blender-3.6"
        / "blender"
    )
    default_blender = bundled_blender if bundled_blender.exists() else Path("/usr/bin/blender")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate CoVeTwin and baseline geometry, semantics, articulation and executability.",
    )
    parser.add_argument(
        "--pred-roots",
        nargs="*",
        type=Path,
        default=[Path("test_demo"), Path("test_demo_new")],
        help=(
            "Prediction roots. test_demo and test_demo_* roots (except "
            "test_demo_new*) use the built-in index mapping; new roots use direct IDs."
        ),
    )
    parser.add_argument(
        "--urdf-anything-roots",
        nargs="*",
        type=Path,
        default=[],
        help="URDF-Anything asset roots (for example urdf_anything_assets and *_new).",
    )
    parser.add_argument(
        "--articulate-roots",
        nargs="*",
        type=Path,
        default=[],
        help="Articulate-Anything result roots (for example results/demo and results/demo_new).",
    )
    parser.add_argument(
        "--physx3d-roots",
        nargs="*",
        type=Path,
        default=[],
        help="PhysX-3D URDF output roots (for example outputs_demo_urdf and *_new_urdf).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset/PhysX_mobility")
    )
    parser.add_argument(
        "--renders-root", type=Path, default=Path("dataset_toolkits/renders_all")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_results/covetwin_metrics")
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        help="Optional sample-name to GT-ID mapping. It overrides the built-in test_demo entries.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Evaluate selected sample IDs, GT IDs, or root:sample pairs (for example test_demo:0).",
    )
    parser.add_argument("--surface-samples", type=int, default=50_000)
    parser.add_argument("--alignment-samples", type=int, default=4_000)
    parser.add_argument("--icp-iterations", type=int, default=20)
    parser.add_argument("--icp-candidates", type=int, default=6)
    parser.add_argument(
        "--alignment",
        choices=["cube_icp", "canonical"],
        default="cube_icp",
        help="cube_icp estimates a rigid frame transform; canonical keeps normalized input frames.",
    )
    parser.add_argument(
        "--fscore-threshold",
        type=float,
        default=0.05,
        help="Surface distance threshold after largest-extent normalization.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-psnr", action="store_true", help="Do not calculate appearance PSNR."
    )
    parser.add_argument(
        "--no-render-missing",
        action="store_true",
        help="Only use already cached predicted views; leave PSNR unavailable when they are missing.",
    )
    parser.add_argument("--force-rerender", action="store_true")
    parser.add_argument(
        "--blender",
        type=Path,
        default=default_blender,
        help=(
            "Blender executable. The bundled Blender 3.6 is preferred because "
            "system Blender 2.82 cannot compile Cycles CUDA kernels for Ada/sm_89 GPUs."
        ),
    )
    parser.add_argument(
        "--render-engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES"
    )
    parser.add_argument("--render-device", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument(
        "--no-render-cpu-fallback",
        action="store_true",
        help=(
            "Do not retry failed Cycles GPU renders on CPU. By default a GPU "
            "failure (for example an unsupported CUDA architecture in an old "
            "Blender build) is retried with Cycles CPU."
        ),
    )
    parser.add_argument("--render-samples", type=int, default=128)
    parser.add_argument(
        "--render-timeout",
        type=int,
        default=1800,
        help="Maximum seconds allowed for one object's Blender render job.",
    )
    parser.add_argument("--skip-executability", action="store_true")
    parser.add_argument(
        "--score-missing-semantics-as-zero",
        action="store_true",
        help=(
            "Treat methods without discrete material/affordance outputs as all-missing "
            "predictions (F1=0) instead of reporting those metrics as N/A."
        ),
    )
    parser.add_argument("--simulation-steps", type=int, default=100)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_seed(base_seed: int, *items: str) -> int:
    digest = hashlib.sha256("::".join(items).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def numeric_sort_key(path: Path) -> tuple[int, Any]:
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def discover_samples(
    args: argparse.Namespace, mapping: dict[str, str]
) -> list[SampleSpec]:
    selected = set(args.only)
    samples: list[SampleSpec] = []
    configured_roots = [
        *(("covetwin", root) for root in args.pred_roots),
        *(("urdf_anything", root) for root in args.urdf_anything_roots),
        *(("articulate_anything", root) for root in args.articulate_roots),
        *(("physx3d", root) for root in args.physx3d_roots),
    ]
    used_set_names: dict[str, int] = {}
    for adapter, root in configured_roots:
        if not root.exists():
            print(f"[WARN] prediction root does not exist: {root}", file=sys.stderr)
            continue
        occurrence = used_set_names.get(root.name, 0)
        used_set_names[root.name] = occurrence + 1
        prediction_set = (
            root.name if occurrence == 0 else f"{root.name}_{adapter}_{occurrence + 1}"
        )
        for sample_dir in sorted(
            (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=numeric_sort_key,
        ):
            gt_id = mapping.get(sample_dir.name, sample_dir.name)
            selectors = {
                sample_dir.name,
                gt_id,
                f"{prediction_set}:{sample_dir.name}",
                f"{adapter}:{sample_dir.name}",
                str(sample_dir),
            }
            if selected and not selectors.intersection(selected):
                continue
            samples.append(
                SampleSpec(adapter, prediction_set, root, sample_dir, gt_id)
            )
    return samples


def mesh_from_loaded(loaded: Any) -> "trimesh.Trimesh":
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError("scene contains no mesh geometry")
        return trimesh.util.concatenate(geometries)
    raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")


def load_mesh_files(paths: Iterable[Path]) -> "trimesh.Trimesh":
    meshes = []
    errors = []
    for path in paths:
        try:
            loaded = trimesh.load(str(path), force="mesh", process=False)
            mesh = mesh_from_loaded(loaded)
            if len(mesh.vertices) and len(mesh.faces):
                meshes.append(mesh)
        except Exception as exc:  # keep evaluating the usable parts
            errors.append(f"{path}: {exc}")
    if not meshes:
        detail = "; ".join(errors[:3])
        raise ValueError(f"no usable triangle meshes; {detail}")
    return trimesh.util.concatenate(meshes)


@dataclass
class VisualAsset:
    path: Path
    transform: np.ndarray
    part_id: int
    link_name: str


@dataclass
class AdaptedAsset:
    adapter: str
    mesh: "trimesh.Trimesh"
    part_meshes: dict[int, "trimesh.Trimesh"]
    pred_data: dict[str, Any]
    urdf_path: Path | None
    render_visuals: list[VisualAsset]
    joints: list[dict[str, Any]]
    semantic_fields: set[str]
    scale_source: str
    warnings: list[str]


def xyz_rpy_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    matrix = trimesh.transformations.euler_matrix(
        float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz"
    )
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def xml_vector(element: ET.Element | None, attribute: str, default: str) -> list[float]:
    text = default if element is None else element.get(attribute, default)
    values = [float(item) for item in str(text).split()]
    if len(values) != 3:
        raise ValueError(f"expected three values for {attribute}, received {text!r}")
    return values


def element_origin_matrix(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    return xyz_rpy_matrix(
        xml_vector(element, "xyz", "0 0 0"),
        xml_vector(element, "rpy", "0 0 0"),
    )


def urdf_link_reference(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return element.get("link") or element.get("name")


def resolve_urdf_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_path.parent / path
    return path.resolve()


def transformed_mesh(path: Path, transform: np.ndarray) -> "trimesh.Trimesh":
    loaded = trimesh.load(str(path), force="mesh", process=False)
    mesh = mesh_from_loaded(loaded).copy()
    mesh.apply_transform(transform)
    return mesh


def concatenate_mesh_dict(
    meshes: dict[int, "trimesh.Trimesh"],
) -> "trimesh.Trimesh":
    usable = [mesh for mesh in meshes.values() if len(mesh.vertices) and len(mesh.faces)]
    if not usable:
        raise ValueError("prediction contains no usable visual meshes")
    return trimesh.util.concatenate(usable)


def parse_urdf_asset(urdf_path: Path, adapter: str) -> AdaptedAsset:
    root = ET.parse(urdf_path).getroot()
    warnings: list[str] = []
    link_elements = list(root.findall("./link"))
    link_names = [str(link.get("name")) for link in link_elements]
    links = set(link_names)

    raw_joints: list[dict[str, Any]] = []
    for index, joint in enumerate(root.findall("./joint")):
        parent = urdf_link_reference(joint.find("parent"))
        child = urdf_link_reference(joint.find("child"))
        if parent not in links or child not in links:
            warnings.append(
                f"joint {joint.get('name', index)} references unknown links: {parent}->{child}"
            )
            continue
        limit = joint.find("limit")
        lower = None if limit is None or limit.get("lower") is None else float(limit.get("lower"))
        upper = None if limit is None or limit.get("upper") is None else float(limit.get("upper"))
        raw_joints.append(
            {
                "name": joint.get("name", f"joint_{index}"),
                "type": str(joint.get("type", "fixed")).lower(),
                "parent": parent,
                "child": child,
                "origin_matrix": element_origin_matrix(joint.find("origin")),
                "axis": xml_vector(joint.find("axis"), "xyz", "1 0 0"),
                "lower": lower,
                "upper": upper,
            }
        )

    # Build a deterministic spanning tree.  Invalid generated URDFs sometimes
    # assign two parents to one link; prefer the articulated edge for metric
    # extraction while preserving the original file for executability testing.
    parent_joint: dict[str, dict[str, Any]] = {}
    for joint in raw_joints:
        child = joint["child"]
        previous = parent_joint.get(child)
        if previous is None:
            parent_joint[child] = joint
        elif previous["type"] == "fixed" and joint["type"] != "fixed":
            parent_joint[child] = joint
            warnings.append(
                f"multiple parents for {child}; selected articulated joint {joint['name']}"
            )
        else:
            warnings.append(
                f"multiple parents for {child}; retained joint {previous['name']}"
            )

    link_world: dict[str, np.ndarray] = {}
    roots = [name for name in link_names if name not in parent_joint]
    for name in roots:
        link_world[name] = np.eye(4, dtype=np.float64)
    for _ in range(max(1, len(link_names))):
        changed = False
        for child, joint in parent_joint.items():
            if child in link_world or joint["parent"] not in link_world:
                continue
            link_world[child] = link_world[joint["parent"]] @ joint["origin_matrix"]
            changed = True
        if not changed:
            break
    for name in link_names:
        if name not in link_world:
            warnings.append(f"unresolved/cyclic link transform for {name}; using identity")
            link_world[name] = np.eye(4, dtype=np.float64)

    visual_links = [
        link for link in link_elements if link.findall("./visual/geometry/mesh")
    ]
    link_to_part = {
        str(link.get("name")): index for index, link in enumerate(visual_links)
    }
    parts = [
        {"label": index, "name": str(link.get("name"))}
        for index, link in enumerate(visual_links)
    ]
    visuals: list[VisualAsset] = []
    part_mesh_lists: dict[int, list["trimesh.Trimesh"]] = {
        index: [] for index in range(len(parts))
    }
    for link in visual_links:
        link_name = str(link.get("name"))
        part_id = link_to_part[link_name]
        for visual in link.findall("./visual"):
            mesh_element = visual.find("./geometry/mesh")
            if mesh_element is None or not mesh_element.get("filename"):
                continue
            path = resolve_urdf_mesh_path(urdf_path, str(mesh_element.get("filename")))
            if not path.exists():
                warnings.append(f"visual mesh does not exist: {path}")
                continue
            scale = xml_vector(mesh_element, "scale", "1 1 1")
            scale_matrix = np.eye(4, dtype=np.float64)
            scale_matrix[:3, :3] = np.diag(np.asarray(scale, dtype=np.float64))
            transform = (
                link_world[link_name]
                @ element_origin_matrix(visual.find("origin"))
                @ scale_matrix
            )
            visuals.append(VisualAsset(path, transform, part_id, link_name))
            try:
                part_mesh_lists[part_id].append(transformed_mesh(path, transform))
            except Exception as exc:
                warnings.append(f"failed to load {path}: {exc}")

    part_meshes = {
        part_id: trimesh.util.concatenate(meshes)
        for part_id, meshes in part_mesh_lists.items()
        if meshes
    }
    mesh = concatenate_mesh_dict(part_meshes)

    children_by_parent: dict[str, list[str]] = {}
    for child, joint in parent_joint.items():
        children_by_parent.setdefault(joint["parent"], []).append(child)

    def subtree_members(child: str) -> set[int]:
        members: set[int] = set()
        stack = [child]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in link_to_part:
                members.add(link_to_part[current])
            stack.extend(children_by_parent.get(current, []))
        return members

    joints: list[dict[str, Any]] = []
    for joint in raw_joints:
        if joint["type"] == "fixed":
            continue
        parent_world = link_world.get(joint["parent"], np.eye(4))
        joint_world = parent_world @ joint["origin_matrix"]
        axis = joint_world[:3, :3] @ np.asarray(joint["axis"], dtype=np.float64)
        joints.append(
            {
                **joint,
                "origin": joint_world[:3, 3].tolist(),
                "axis_world": axis.tolist(),
                "members": subtree_members(joint["child"]),
            }
        )

    pred_data: dict[str, Any] = {"parts": parts, "group_info": {}}
    return AdaptedAsset(
        adapter=adapter,
        mesh=mesh,
        part_meshes=part_meshes,
        pred_data=pred_data,
        urdf_path=urdf_path,
        render_visuals=visuals,
        joints=joints,
        semantic_fields=set(),
        scale_source="unreported",
        warnings=warnings,
    )


def groups_from_urdf_joints(
    joints: list[dict[str, Any]], pred_mesh_extent: float
) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    code_map = {
        "prismatic": "B",
        "revolute": "C",
        "continuous": "C",
        "spherical": "D",
        "floating": "A",
    }
    for index, joint in enumerate(joints, start=1):
        code = code_map.get(joint["type"])
        if code is None:
            continue
        axis = list(joint["axis_world"])
        origin = list(joint["origin"])
        if code == "C":
            if joint["type"] == "continuous":
                limits = [-1.0, 1.0]
            else:
                lower = 0.0 if joint["lower"] is None else float(joint["lower"])
                upper = 0.0 if joint["upper"] is None else float(joint["upper"])
                limits = [lower / math.pi, upper / math.pi]
            params = axis + origin + limits
        elif code == "B":
            scale = max(float(pred_mesh_extent), 1e-12)
            lower = 0.0 if joint["lower"] is None else float(joint["lower"])
            upper = 0.0 if joint["upper"] is None else float(joint["upper"])
            params = axis + origin + [lower / scale, upper / scale]
        else:
            params = axis + origin
        groups[str(index)] = [
            sorted(int(item) for item in joint["members"]),
            "0",
            params,
            code,
        ]
    return groups


def groups_from_physx3d_metadata(metadata: dict[str, Any]) -> dict[str, list[Any]]:
    """Preserve PhysX-3D's native A/B/C/D label before URDF approximation."""
    groups: dict[str, list[Any]] = {}
    for group_id, raw in metadata.get("group_info", {}).items():
        if str(group_id) == "0" or not isinstance(raw, dict):
            continue
        code = str(raw.get("joint_type", "E")).upper()
        if code == "E":
            continue
        axis = [float(item) for item in raw.get("axis", [0, 0, 1])]
        origin = [float(item) for item in raw.get("origin", [0, 0, 0])]
        limits = [float(item) for item in raw.get("range", [0, 0])]
        params = axis + origin
        if code in ("B", "C"):
            params += limits
        groups[str(group_id)] = [
            [int(group_id)],
            str(raw.get("parent", "0")),
            params,
            code,
        ]
    return groups


def identity_visuals(paths: Iterable[Path]) -> list[VisualAsset]:
    return [
        VisualAsset(Path(path).resolve(), np.eye(4, dtype=np.float64), index, str(index))
        for index, path in enumerate(paths)
    ]


def adapt_covetwin_sample(sample_dir: Path) -> AdaptedAsset:
    pred_data = load_json(sample_dir / "basic_info.json")
    paths = prediction_mesh_paths(sample_dir)
    if not paths:
        raise ValueError(f"no prediction OBJ files under {sample_dir / 'objs'}")
    part_meshes: dict[int, "trimesh.Trimesh"] = {}
    visuals: list[VisualAsset] = []
    for index, part in enumerate(pred_data.get("parts", [])):
        try:
            part_id = int(part.get("label", index))
        except (TypeError, ValueError):
            part_id = index
        candidates = sorted((sample_dir / "objs" / str(part_id)).glob("*.obj"))
        if not candidates and index < len(paths):
            candidates = [paths[index]]
        if candidates:
            part_meshes[part_id] = load_mesh_files(candidates)
            visuals.extend(
                VisualAsset(path.resolve(), np.eye(4), part_id, f"l_{part_id}")
                for path in candidates
            )
    if not part_meshes:
        part_meshes = {0: load_mesh_files(paths)}
        visuals = identity_visuals(paths)
    return AdaptedAsset(
        adapter="covetwin",
        mesh=concatenate_mesh_dict(part_meshes),
        part_meshes=part_meshes,
        pred_data=pred_data,
        urdf_path=(sample_dir / "basic.urdf"),
        render_visuals=visuals,
        joints=[],
        semantic_fields={"material", "priority_rank"},
        scale_source="generated_mesh",
        warnings=[],
    )


def adapt_prediction(spec: SampleSpec) -> AdaptedAsset:
    if spec.adapter == "covetwin":
        return adapt_covetwin_sample(spec.sample_dir)
    if spec.adapter == "urdf_anything":
        urdf_path = spec.sample_dir / "mesh_reconstruction" / "mobility.urdf"
        asset = parse_urdf_asset(urdf_path, spec.adapter)
        prediction_path = spec.sample_dir / "prediction.json"
        if prediction_path.exists():
            prediction = load_json(prediction_path)
            link_names = prediction.get("pred_answers", {}).get("links", {})
            for part in asset.pred_data.get("parts", []):
                link_name = str(part.get("name", ""))
                if link_name in link_names:
                    part["name"] = link_names[link_name]
        asset.scale_source = "inherited_input_mesh"
        return asset
    if spec.adapter == "articulate_anything":
        candidates = [
            spec.sample_dir / "joint_actor" / "iter_0" / "seed_0" / "mobility.urdf",
            spec.sample_dir / "link_placement" / "iter_0" / "seed_0" / "mobility.urdf",
        ]
        urdf_path = next((path for path in candidates if path.exists()), None)
        if urdf_path is None:
            raise FileNotFoundError(f"no Articulate-Anything mobility.urdf under {spec.sample_dir}")
        asset = parse_urdf_asset(urdf_path, spec.adapter)
        asset.scale_source = "retrieved_mesh"
        return asset
    if spec.adapter == "physx3d":
        urdf_path = spec.sample_dir / "urdf_export" / "mobility.urdf"
        asset = parse_urdf_asset(urdf_path, spec.adapter)
        texture_glb = spec.sample_dir / "texture.glb"
        if texture_glb.exists():
            asset.mesh = mesh_from_loaded(
                trimesh.load(str(texture_glb), force="mesh", process=False)
            )
            asset.render_visuals = [
                VisualAsset(texture_glb.resolve(), np.eye(4), 0, "texture_mesh")
            ]
        metadata_path = spec.sample_dir / "urdf_export" / "physx3d_kinematic.json"
        if metadata_path.exists():
            asset.pred_data["_physx3d_metadata"] = load_json(metadata_path)
        asset.scale_source = "generated_normalized_mesh"
        return asset
    raise ValueError(f"unknown prediction adapter: {spec.adapter}")


def prediction_mesh_paths(sample_dir: Path) -> list[Path]:
    return sorted(sample_dir.glob("objs/**/*.obj"))


def gt_mesh_paths(
    dataset_root: Path, gt_id: str, gt_data: dict[str, Any]
) -> list[Path]:
    obj_dir = dataset_root / "partseg" / gt_id / "objs"
    paths: list[Path] = []
    for part in gt_data.get("parts", []):
        refs = part.get("obj", [])
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            name = str(ref)
            if not name.lower().endswith(".obj"):
                name += ".obj"
            path = obj_dir / name
            if path.exists() and path not in paths:
                paths.append(path)
    return paths or sorted(obj_dir.glob("*.obj"))


def gt_part_meshes(
    dataset_root: Path, gt_id: str, gt_data: dict[str, Any]
) -> dict[int, "trimesh.Trimesh"]:
    obj_dir = dataset_root / "partseg" / gt_id / "objs"
    result: dict[int, "trimesh.Trimesh"] = {}
    for index, part in enumerate(gt_data.get("parts", [])):
        try:
            part_id = int(part.get("label", index))
        except (TypeError, ValueError):
            part_id = index
        refs = part.get("obj", [])
        if isinstance(refs, str):
            refs = [refs]
        paths = []
        for ref in refs:
            name = str(ref)
            if not name.lower().endswith(".obj"):
                name += ".obj"
            path = obj_dir / name
            if path.exists():
                paths.append(path)
        if paths:
            result[part_id] = load_mesh_files(paths)
    return result


def mesh_normalization(mesh: "trimesh.Trimesh") -> Normalization:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"degenerate mesh bounding box: {bounds.tolist()}")
    return Normalization(center=(bounds[0] + bounds[1]) / 2.0, scale=scale)


def sample_surface(mesh: "trimesh.Trimesh", count: int, seed: int) -> np.ndarray:
    count = max(100, int(count))
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("surface sampler produced non-finite points")
    return points


def cube_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError(f"invalid zero-length vector: {value.tolist()}")
    return value / norm


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def symmetric_l1(source: np.ndarray, target: np.ndarray) -> float:
    target_tree = cKDTree(target)
    source_tree = cKDTree(source)
    d_st = target_tree.query(source, workers=-1)[0]
    d_ts = source_tree.query(target, workers=-1)[0]
    return float(np.mean(d_st) + np.mean(d_ts))


def run_trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    iterations: int,
    trim_fraction: float = 0.90,
) -> Alignment:
    target_tree = cKDTree(target)
    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    transformed = source @ rotation.T + translation
    previous = math.inf
    for _ in range(max(1, iterations)):
        distances, indices = target_tree.query(transformed, workers=-1)
        cutoff = np.quantile(distances, trim_fraction)
        keep = distances <= cutoff
        if int(keep.sum()) < 3:
            break
        delta_rotation, delta_translation = kabsch(
            transformed[keep], target[indices[keep]]
        )
        transformed = transformed @ delta_rotation.T + delta_translation
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        error = float(np.mean(distances[keep]))
        if abs(previous - error) < 1e-7:
            break
        previous = error
    return Alignment(rotation, translation, symmetric_l1(transformed, target))


def estimate_alignment(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Alignment:
    if args.alignment == "canonical":
        return Alignment(np.eye(3), np.zeros(3), symmetric_l1(pred_points, gt_points))

    rng = np.random.default_rng(seed)
    count = min(args.alignment_samples, len(pred_points), len(gt_points))
    pred = pred_points[rng.choice(len(pred_points), count, replace=False)]
    gt = gt_points[rng.choice(len(gt_points), count, replace=False)]

    initial_scores: list[tuple[float, np.ndarray]] = []
    gt_tree = cKDTree(gt)
    for rotation in cube_rotations():
        translation = gt.mean(axis=0) - rotation @ pred.mean(axis=0)
        transformed = pred @ rotation.T + translation
        score = float(np.mean(gt_tree.query(transformed, workers=-1)[0]))
        initial_scores.append((score, rotation))
    initial_scores.sort(key=lambda item: item[0])

    candidates = []
    for _, rotation in initial_scores[: max(1, args.icp_candidates)]:
        candidates.append(run_trimmed_icp(pred, gt, rotation, args.icp_iterations))
    return min(candidates, key=lambda item: item.score)


def geometry_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    alignment: Alignment,
    threshold: float,
) -> dict[str, float]:
    pred_aligned = alignment.points(pred_points)
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_aligned)
    pred_to_gt = gt_tree.query(pred_aligned, workers=-1)[0]
    gt_to_pred = pred_tree.query(gt_points, workers=-1)[0]
    precision = float(np.mean(pred_to_gt <= threshold))
    recall = float(np.mean(gt_to_pred <= threshold))
    fscore = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    cd_l2 = float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2))
    return {
        "chamfer_l2": cd_l2,
        "chamfer_l2_x1e3": cd_l2 * 1000.0,
        "chamfer_l1": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "fscore": fscore,
        "fscore_precision": precision,
        "fscore_recall": recall,
        "fscore_threshold": float(threshold),
        "surface_samples": int(min(len(pred_points), len(gt_points))),
        "alignment_score": float(alignment.score),
        "alignment_rotation_pred_to_gt": alignment.rotation.tolist(),
        "alignment_translation_pred_to_gt": alignment.translation.tolist(),
    }


def geometric_part_matches(
    pred_parts: dict[int, "trimesh.Trimesh"],
    gt_parts: dict[int, "trimesh.Trimesh"],
    pred_norm: Normalization,
    gt_norm: Normalization,
    alignment: Alignment,
    seed: int,
    samples_per_part: int = 1500,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Match generic URDF links to GT parts by aligned surface proximity."""
    if not pred_parts or not gt_parts:
        return {}, []
    pred_ids = sorted(pred_parts)
    gt_ids = sorted(gt_parts)
    pred_points = []
    gt_points = []
    for offset, part_id in enumerate(pred_ids):
        points = sample_surface(
            pred_parts[part_id], samples_per_part, seed + 17 * offset
        )
        pred_points.append(alignment.points(pred_norm.points(points)))
    for offset, part_id in enumerate(gt_ids):
        points = sample_surface(gt_parts[part_id], samples_per_part, seed + 1009 + 17 * offset)
        gt_points.append(gt_norm.points(points))

    costs = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    for row, source in enumerate(pred_points):
        for column, target in enumerate(gt_points):
            costs[row, column] = symmetric_l1(source, target)
    rows, columns = linear_sum_assignment(costs)
    matches: dict[int, int] = {}
    details = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        pred_id = pred_ids[row]
        gt_id = gt_ids[column]
        matches[pred_id] = gt_id
        details.append(
            {
                "pred_part_id": pred_id,
                "gt_part_id": gt_id,
                "method": "aligned_part_chamfer_hungarian",
                "cost": float(costs[row, column]),
            }
        )
    return matches, details


def dimension_values(value: Any) -> list[float]:
    if value is None:
        return []
    text = str(value).splitlines()[0]
    values = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)]
    return values[:3]


def scale_metrics(pred_data: dict[str, Any], gt_data: dict[str, Any]) -> dict[str, Any]:
    pred = dimension_values(pred_data.get("dimension"))
    gt = dimension_values(gt_data.get("dimension"))
    if len(pred) < 3 or len(gt) < 3:
        return {
            "available": False,
            "reason": "dimension must contain three numeric values",
        }
    pred_sorted = np.sort(np.asarray(pred, dtype=np.float64))[::-1]
    gt_sorted = np.sort(np.asarray(gt, dtype=np.float64))[::-1]
    error = float(np.linalg.norm(pred_sorted - gt_sorted))
    return {
        "available": True,
        "pred_dimension_cm": pred,
        "gt_dimension_cm": gt,
        "absolute_scale_error_cm": error,
        "max_dimension_absolute_error_cm": float(abs(pred_sorted[0] - gt_sorted[0])),
        "relative_scale_error": float(error / max(np.linalg.norm(gt_sorted), 1e-12)),
        "pred_max_dimension_m": float(pred_sorted[0] / 100.0),
        "gt_max_dimension_m": float(gt_sorted[0] / 100.0),
    }


def mesh_scale_metrics(
    pred_mesh: "trimesh.Trimesh",
    gt_mesh: "trimesh.Trimesh",
    gt_data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Coordinate-scale proxy for methods that do not report metric dimensions.

    GT mesh units are calibrated using the annotated largest dimension, then
    the same unit-to-centimeter conversion is applied to prediction extents.
    This keeps the calculation identical across adapters while explicitly
    recording that it is a mesh-coordinate proxy rather than a reported scale.
    """
    gt_dimension = dimension_values(gt_data.get("dimension"))
    if len(gt_dimension) < 3:
        return {"available": False, "reason": "GT dimension is unavailable"}
    pred_extents = np.sort(np.asarray(pred_mesh.extents, dtype=np.float64))[::-1]
    gt_extents = np.sort(np.asarray(gt_mesh.extents, dtype=np.float64))[::-1]
    gt_cm = np.sort(np.asarray(gt_dimension, dtype=np.float64))[::-1]
    if gt_extents[0] <= 1e-12 or pred_extents[0] <= 1e-12:
        return {"available": False, "reason": "degenerate mesh extents"}
    centimeters_per_coordinate_unit = float(gt_cm[0] / gt_extents[0])
    pred_cm = pred_extents * centimeters_per_coordinate_unit
    error = float(np.linalg.norm(pred_cm - gt_cm))
    return {
        "available": True,
        "protocol": "GT-largest-dimension calibrated raw mesh-coordinate extents",
        "source": source,
        "pred_dimension_cm": pred_cm.tolist(),
        "gt_dimension_cm": gt_cm.tolist(),
        "pred_mesh_extents": pred_extents.tolist(),
        "gt_mesh_extents": gt_extents.tolist(),
        "absolute_scale_error_cm": error,
        "max_dimension_absolute_error_cm": float(abs(pred_cm[0] - gt_cm[0])),
        "relative_scale_error": float(error / max(np.linalg.norm(gt_cm), 1e-12)),
        "pred_max_dimension_m": float(pred_cm[0] / 100.0),
        "gt_max_dimension_m": float(gt_cm[0] / 100.0),
    }


def canonical_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split()) or "__missing__"


def canonical_rank(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "__missing__"
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".8g")
    except (TypeError, ValueError):
        return canonical_text(value)


def part_label(part: dict[str, Any], index: int) -> str:
    return str(part.get("label", index))


def match_parts(
    pred_parts: list[dict[str, Any]], gt_parts: list[dict[str, Any]]
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    matches: dict[int, int] = {}
    details: list[dict[str, Any]] = []
    pred_by_label: dict[str, list[int]] = {}
    gt_by_label: dict[str, list[int]] = {}
    for index, part in enumerate(pred_parts):
        pred_by_label.setdefault(part_label(part, index), []).append(index)
    for index, part in enumerate(gt_parts):
        gt_by_label.setdefault(part_label(part, index), []).append(index)

    for label in sorted(set(pred_by_label).intersection(gt_by_label)):
        if len(pred_by_label[label]) == len(gt_by_label[label]) == 1:
            pred_index = pred_by_label[label][0]
            gt_index = gt_by_label[label][0]
            matches[pred_index] = gt_index
            details.append(
                {
                    "pred_index": pred_index,
                    "gt_index": gt_index,
                    "method": "label",
                    "score": 1.0,
                }
            )

    remaining_pred = [i for i in range(len(pred_parts)) if i not in matches]
    used_gt = set(matches.values())
    remaining_gt = [i for i in range(len(gt_parts)) if i not in used_gt]
    if remaining_pred and remaining_gt:
        costs = np.zeros((len(remaining_pred), len(remaining_gt)), dtype=np.float64)
        for row, pred_index in enumerate(remaining_pred):
            pred_name = canonical_text(pred_parts[pred_index].get("name", ""))
            for column, gt_index in enumerate(remaining_gt):
                gt_name = canonical_text(gt_parts[gt_index].get("name", ""))
                similarity = difflib.SequenceMatcher(None, pred_name, gt_name).ratio()
                costs[row, column] = 1.0 - similarity
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows.tolist(), columns.tolist()):
            pred_index = remaining_pred[row]
            gt_index = remaining_gt[column]
            matches[pred_index] = gt_index
            details.append(
                {
                    "pred_index": pred_index,
                    "gt_index": gt_index,
                    "method": "name_hungarian",
                    "score": float(1.0 - costs[row, column]),
                }
            )
    return matches, sorted(details, key=lambda item: item["gt_index"])


def f1_scores(records: list[dict[str, str]]) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    labels = sorted(
        {item["true"] for item in records if item["true"] != "__spurious__"}
    )
    f1_values = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(item["true"] == label and item["pred"] == label for item in records)
        fp = sum(item["true"] != label and item["pred"] == label for item in records)
        fn = sum(item["true"] == label and item["pred"] != label for item in records)
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    denominator = 2 * total_tp + total_fp + total_fn
    micro = 0.0 if denominator == 0 else 2.0 * total_tp / denominator
    return float(np.mean(f1_values)) if f1_values else None, float(micro)


def semantic_records(
    pred_parts: list[dict[str, Any]],
    gt_parts: list[dict[str, Any]],
    matches: dict[int, int],
    field: str,
) -> list[dict[str, str]]:
    reverse = {gt_index: pred_index for pred_index, gt_index in matches.items()}
    records = []
    for gt_index, gt_part in enumerate(gt_parts):
        pred_index = reverse.get(gt_index)
        normalize = canonical_rank if field == "priority_rank" else canonical_text
        true_label = normalize(gt_part.get(field))
        pred_label = (
            "__missing__"
            if pred_index is None
            else normalize(pred_parts[pred_index].get(field))
        )
        records.append({"true": true_label, "pred": pred_label})
    for pred_index, pred_part in enumerate(pred_parts):
        if pred_index not in matches:
            normalize = canonical_rank if field == "priority_rank" else canonical_text
            records.append(
                {"true": "__spurious__", "pred": normalize(pred_part.get(field))}
            )
    return records


def semantic_metrics(
    pred_data: dict[str, Any],
    gt_data: dict[str, Any],
    supported_fields: set[str] | None = None,
) -> tuple[dict[str, Any], dict[int, int]]:
    if supported_fields is None:
        supported_fields = {"material", "priority_rank"}
    pred_parts = list(pred_data.get("parts", []))
    gt_parts = list(gt_data.get("parts", []))
    matches, match_details = match_parts(pred_parts, gt_parts)
    material = (
        semantic_records(pred_parts, gt_parts, matches, "material")
        if "material" in supported_fields
        else []
    )
    affordance = (
        semantic_records(pred_parts, gt_parts, matches, "priority_rank")
        if "priority_rank" in supported_fields
        else []
    )
    material_macro, material_micro = f1_scores(material)
    affordance_macro, affordance_micro = f1_scores(affordance)
    coverage = len(matches) / max(len(pred_parts), len(gt_parts), 1)
    group_part_matches: dict[int, int] = {}
    for pred_index, gt_index in matches.items():
        try:
            pred_id = int(pred_parts[pred_index].get("label", pred_index))
            gt_id = int(gt_parts[gt_index].get("label", gt_index))
        except (TypeError, ValueError):
            pred_id, gt_id = pred_index, gt_index
        group_part_matches[pred_id] = gt_id
    return (
        {
            "available": bool(pred_parts and gt_parts),
            "material_available": "material" in supported_fields,
            "material_reason": None
            if "material" in supported_fields
            else "prediction method does not export discrete material labels",
            "affordance_available": "priority_rank" in supported_fields,
            "affordance_reason": None
            if "priority_rank" in supported_fields
            else "prediction method does not export discrete affordance classes",
            "material_macro_f1": material_macro,
            "material_micro_f1": material_micro,
            "affordance_macro_f1": affordance_macro,
            "affordance_micro_f1": affordance_micro,
            "part_match_coverage": float(coverage),
            "num_pred_parts": len(pred_parts),
            "num_gt_parts": len(gt_parts),
            "num_matched_parts": len(matches),
            "part_matches": match_details,
            "material_records": material,
            "affordance_records": affordance,
        },
        group_part_matches,
    )


def parse_groups(data: dict[str, Any]) -> list[Group]:
    groups = []
    for group_id, raw in data.get("group_info", {}).items():
        if str(group_id) == "0" or not isinstance(raw, list) or len(raw) < 4:
            continue
        try:
            members = {int(item) for item in raw[0]}
            params = [float(item) for item in raw[2]]
        except (TypeError, ValueError):
            continue
        groups.append(
            Group(str(group_id), members, str(raw[1]), params, str(raw[3]).upper())
        )
    return groups


def group_matches(
    pred_groups: list[Group], gt_groups: list[Group], part_matches: dict[int, int]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not pred_groups or not gt_groups:
        return [], list(range(len(pred_groups))), list(range(len(gt_groups)))
    cost = np.ones((len(pred_groups), len(gt_groups)), dtype=np.float64)
    similarities = np.zeros_like(cost)
    for pred_index, pred_group in enumerate(pred_groups):
        mapped = {
            part_matches[item] for item in pred_group.members if item in part_matches
        }
        for gt_index, gt_group in enumerate(gt_groups):
            union = mapped | gt_group.members
            similarity = len(mapped & gt_group.members) / len(union) if union else 1.0
            similarities[pred_index, gt_index] = similarity
            cost[pred_index, gt_index] = 1.0 - similarity
    rows, columns = linear_sum_assignment(cost)
    matches = []
    used_pred = set()
    used_gt = set()
    for pred_index, gt_index in zip(rows.tolist(), columns.tolist()):
        similarity = float(similarities[pred_index, gt_index])
        if similarity <= 0.0:
            continue
        matches.append((pred_index, gt_index, similarity))
        used_pred.add(pred_index)
        used_gt.add(gt_index)
    return (
        matches,
        [i for i in range(len(pred_groups)) if i not in used_pred],
        [i for i in range(len(gt_groups)) if i not in used_gt],
    )


def group_dofs(group: Group) -> list[dict[str, Any]]:
    p = group.params
    if group.code == "B" and len(p) >= 8:
        return [
            {
                "type": "prismatic",
                "axis": p[:3],
                "origin": p[3:6],
                "limits": p[6:8],
                "slot": "B",
            }
        ]
    if group.code == "C" and len(p) >= 8:
        return [
            {
                "type": "revolute",
                "axis": p[:3],
                "origin": p[3:6],
                "limits": p[6:8],
                "slot": "C",
            }
        ]
    if group.code == "D" and len(p) >= 6:
        return [
            {
                "type": "ball",
                "axis": None,
                "origin": p[3:6],
                "limits": None,
                "slot": "D",
            }
        ]
    if group.code == "CB" and len(p) >= 16:
        return [
            {
                "type": "revolute",
                "axis": p[:3],
                "origin": p[3:6],
                "limits": p[6:8],
                "slot": "C",
            },
            {
                "type": "prismatic",
                "axis": p[8:11],
                "origin": p[3:6],
                "limits": p[14:16],
                "slot": "B",
            },
        ]
    return []


def transform_origin(
    origin: Sequence[float], normalization: Normalization, alignment: Alignment | None
) -> np.ndarray:
    point = normalization.points(np.asarray(origin, dtype=np.float64)[None, :])[0]
    return point if alignment is None else alignment.points(point[None, :])[0]


def axis_angle(pred_axis: np.ndarray, gt_axis: np.ndarray) -> float:
    cosine = float(
        np.clip(
            abs(np.dot(normalize_vector(pred_axis), normalize_vector(gt_axis))),
            -1.0,
            1.0,
        )
    )
    return float(math.acos(cosine))


def revolute_origin_distance(
    pred_origin: np.ndarray,
    gt_origin: np.ndarray,
    pred_axis: np.ndarray,
    gt_axis: np.ndarray,
) -> float:
    pred_axis = normalize_vector(pred_axis)
    gt_axis = normalize_vector(gt_axis)
    delta = pred_origin - gt_origin
    cross = np.cross(pred_axis, gt_axis)
    norm = float(np.linalg.norm(cross))
    if norm <= 1e-8:
        return float(np.linalg.norm(np.cross(delta, gt_axis)))
    return float(abs(np.dot(delta, cross)) / norm)


def is_continuous_limits(limits: Sequence[float]) -> bool:
    return (
        len(limits) >= 2
        and abs(float(limits[0]) + 1.0) < 1e-6
        and abs(float(limits[1]) - 1.0) < 1e-6
    )


def dof_metrics(
    pred: dict[str, Any],
    gt: dict[str, Any],
    pred_norm: Normalization,
    gt_norm: Normalization,
    alignment: Alignment,
    pred_max_m: float,
    gt_max_m: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": gt["type"], "slot": gt["slot"]}
    pred_origin = transform_origin(pred["origin"], pred_norm, alignment)
    gt_origin = transform_origin(gt["origin"], gt_norm, None)

    if gt["axis"] is None:
        result["origin_error_m"] = float(
            np.linalg.norm(pred_origin - gt_origin) * gt_max_m
        )
        return result

    pred_axis = alignment.vector(pred["axis"])
    gt_axis = normalize_vector(gt["axis"])
    angle = axis_angle(pred_axis, gt_axis)
    result["axis_error_rad"] = angle
    result["axis_error_deg"] = math.degrees(angle)
    if gt["type"] == "revolute":
        origin_error = revolute_origin_distance(
            pred_origin, gt_origin, pred_axis, gt_axis
        )
    else:
        origin_error = float(np.linalg.norm(pred_origin - gt_origin))
    result["origin_error_m"] = origin_error * gt_max_m

    pred_limits = pred.get("limits")
    gt_limits = gt.get("limits")
    if pred_limits is None or gt_limits is None:
        return result
    if gt["type"] == "revolute":
        pred_continuous = is_continuous_limits(pred_limits)
        gt_continuous = is_continuous_limits(gt_limits)
        result["pred_continuous"] = pred_continuous
        result["gt_continuous"] = gt_continuous
        result["continuous_accuracy"] = float(pred_continuous == gt_continuous)
        if pred_continuous or gt_continuous:
            return result
        pred_span = (float(pred_limits[1]) - float(pred_limits[0])) * math.pi
        gt_span = (float(gt_limits[1]) - float(gt_limits[0])) * math.pi
        result["motion_range_error"] = float(
            np.linalg.norm(pred_axis * pred_span - gt_axis * gt_span)
        )
        result["motion_range_unit"] = "rad"
    elif gt["type"] == "prismatic":
        pred_span = (float(pred_limits[1]) - float(pred_limits[0])) * pred_max_m
        gt_span = (float(gt_limits[1]) - float(gt_limits[0])) * gt_max_m
        result["motion_range_error"] = float(
            np.linalg.norm(pred_axis * pred_span - gt_axis * gt_span)
        )
        result["motion_range_unit"] = "m"
    return result


def mean_present(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(item[field])
        for item in records
        if item.get(field) is not None and math.isfinite(float(item[field]))
    ]
    return float(np.mean(values)) if values else None


def articulation_metrics(
    pred_data: dict[str, Any],
    gt_data: dict[str, Any],
    part_matches: dict[int, int],
    pred_norm: Normalization,
    gt_norm: Normalization,
    alignment: Alignment,
    scale: dict[str, Any],
) -> dict[str, Any]:
    pred_groups = parse_groups(pred_data)
    gt_groups = parse_groups(gt_data)
    matches, unmatched_pred, unmatched_gt = group_matches(
        pred_groups, gt_groups, part_matches
    )
    denominator = max(len(pred_groups), len(gt_groups))
    correct = sum(pred_groups[p].code == gt_groups[g].code for p, g, _ in matches)

    group_records = []
    joint_records = []
    pred_max_m = float(scale.get("pred_max_dimension_m", 1.0))
    gt_max_m = float(scale.get("gt_max_dimension_m", 1.0))
    for pred_index, gt_index, overlap in matches:
        pred_group = pred_groups[pred_index]
        gt_group = gt_groups[gt_index]
        type_correct = pred_group.code == gt_group.code
        group_record = {
            "pred_group_id": pred_group.group_id,
            "gt_group_id": gt_group.group_id,
            "pred_type": pred_group.code,
            "gt_type": gt_group.code,
            "pred_type_name": GROUP_TYPE_NAMES.get(pred_group.code, pred_group.code),
            "gt_type_name": GROUP_TYPE_NAMES.get(gt_group.code, gt_group.code),
            "child_part_iou": overlap,
            "type_correct": bool(type_correct),
        }
        group_records.append(group_record)
        if not type_correct:
            continue
        pred_dofs = group_dofs(pred_group)
        gt_dofs = group_dofs(gt_group)
        pred_by_slot = {item["slot"]: item for item in pred_dofs}
        for gt_dof in gt_dofs:
            pred_dof = pred_by_slot.get(gt_dof["slot"])
            if pred_dof is None or pred_dof["type"] != gt_dof["type"]:
                continue
            record = dof_metrics(
                pred_dof,
                gt_dof,
                pred_norm,
                gt_norm,
                alignment,
                pred_max_m,
                gt_max_m,
            )
            record.update(
                {"pred_group_id": pred_group.group_id, "gt_group_id": gt_group.group_id}
            )
            joint_records.append(record)

    for index in unmatched_gt:
        group_records.append(
            {
                "pred_group_id": None,
                "gt_group_id": gt_groups[index].group_id,
                "pred_type": "__missing__",
                "gt_type": gt_groups[index].code,
                "child_part_iou": 0.0,
                "type_correct": False,
            }
        )
    for index in unmatched_pred:
        group_records.append(
            {
                "pred_group_id": pred_groups[index].group_id,
                "gt_group_id": None,
                "pred_type": pred_groups[index].code,
                "gt_type": "__spurious__",
                "child_part_iou": 0.0,
                "type_correct": False,
            }
        )

    revolute = [
        item for item in joint_records if item.get("motion_range_unit") == "rad"
    ]
    prismatic = [item for item in joint_records if item.get("motion_range_unit") == "m"]
    return {
        "available": True,
        "joint_type_accuracy": None
        if denominator == 0
        else float(correct / denominator),
        "joint_type_correct": int(correct),
        "joint_type_denominator": int(denominator),
        "num_pred_groups": len(pred_groups),
        "num_gt_groups": len(gt_groups),
        "num_matched_groups": len(matches),
        "axis_error_rad": mean_present(joint_records, "axis_error_rad"),
        "axis_error_deg": mean_present(joint_records, "axis_error_deg"),
        "origin_error_m": mean_present(joint_records, "origin_error_m"),
        "motion_range_error": mean_present(joint_records, "motion_range_error"),
        "motion_range_error_note": "Mean of native per-joint errors; inspect type-specific rad/m values when types are mixed.",
        "revolute_motion_range_error_rad": mean_present(revolute, "motion_range_error"),
        "prismatic_motion_range_error_m": mean_present(prismatic, "motion_range_error"),
        "num_axis_pairs": sum(
            item.get("axis_error_rad") is not None for item in joint_records
        ),
        "num_origin_pairs": sum(
            item.get("origin_error_m") is not None for item in joint_records
        ),
        "num_motion_range_pairs": sum(
            item.get("motion_range_error") is not None for item in joint_records
        ),
        "group_records": group_records,
        "joint_records": joint_records,
    }


def psnr_value(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None
) -> float:
    difference = (first.astype(np.float64) - second.astype(np.float64)) ** 2
    if mask is not None:
        if not np.any(mask):
            return 100.0
        difference = difference[mask]
    mse = float(np.mean(difference))
    return 100.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def read_composited_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    return rgb, alpha[..., 0]


def render_manifest_matches(
    output_dir: Path,
    sample_dir: Path,
    camera_json: Path,
    args: argparse.Namespace,
    alignment: Alignment,
    resolution: int,
    frame_count: int,
    render_visuals: list[VisualAsset],
    render_device: str | None = None,
) -> bool:
    manifest_path = output_dir / "render_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return False
    expected = {
        "object_root": str(sample_dir.resolve()),
        "camera_json": str(camera_json.resolve()),
        "engine": args.render_engine,
        "device": render_device or args.render_device,
        "samples": args.render_samples,
        "resolution": resolution,
        "frame_count": frame_count,
        "render_visuals": [
            {
                "path": str(item.path.resolve()),
                "transform": np.asarray(item.transform, dtype=np.float64).tolist(),
            }
            for item in render_visuals
        ],
        "alignment": {
            "rotation": alignment.rotation.tolist(),
            "translation": alignment.translation.tolist(),
        },
    }
    return all(manifest.get(key) == value for key, value in expected.items()) and all(
        (output_dir / f"{index:03d}.png").exists() for index in range(frame_count)
    )


def ensure_prediction_renders(
    sample_dir: Path,
    gt_render_dir: Path,
    cache_dir: Path,
    alignment: Alignment,
    args: argparse.Namespace,
    render_visuals: list[VisualAsset],
) -> tuple[bool, str | None]:
    camera_json = gt_render_dir / "transforms.json"
    camera_data = load_json(camera_json)
    frames = camera_data.get("frames", [])
    if not frames:
        return False, "GT transforms.json has no frames"
    first_gt = gt_render_dir / frames[0]["file_path"]
    with Image.open(first_gt) as image:
        resolution = int(image.size[0])
    if not args.force_rerender:
        cache_devices = [args.render_device]
        if (
            args.render_engine == "CYCLES"
            and args.render_device == "GPU"
            and not args.no_render_cpu_fallback
        ):
            # A CPU fallback render is just as valid for PSNR and should remain
            # reusable on later runs that still request GPU first.
            cache_devices.append("CPU")
        for cached_device in cache_devices:
            if render_manifest_matches(
                cache_dir,
                sample_dir,
                camera_json,
                args,
                alignment,
                resolution,
                len(frames),
                render_visuals,
                render_device=cached_device,
            ):
                return True, None
    if args.no_render_missing:
        return (
            False,
            "prediction render cache is missing or stale and --no-render-missing was set",
        )
    if not args.blender.exists() and shutil.which(str(args.blender)) is None:
        return False, f"Blender executable not found: {args.blender}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    alignment_json = cache_dir / "alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "rotation": alignment.rotation.tolist(),
                "translation": alignment.translation.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mesh_manifest = cache_dir / "mesh_manifest.json"
    mesh_manifest.write_text(
        json.dumps(
            {
                "object_root": str(sample_dir.resolve()),
                "visuals": [
                    {
                        "path": str(item.path.resolve()),
                        "transform": np.asarray(item.transform, dtype=np.float64).tolist(),
                    }
                    for item in render_visuals
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    worker = WORKER_DIR / "render_pred_views.py"
    def run_blender(device: str, log_path: Path, force: bool) -> tuple[bool, str | None]:
        command = [
            str(args.blender),
            "-noaudio",
            "-b",
            "--python",
            str(worker),
            "--",
            "--object-root",
            str(sample_dir.resolve()),
            "--mesh-manifest",
            str(mesh_manifest.resolve()),
            "--camera-json",
            str(camera_json.resolve()),
            "--alignment-json",
            str(alignment_json.resolve()),
            "--output-dir",
            str(cache_dir.resolve()),
            "--engine",
            args.render_engine,
            "--device",
            device,
            "--samples",
            str(args.render_samples),
            "--resolution",
            str(resolution),
        ]
        if force:
            command.append("--force")
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=max(1, args.render_timeout),
                )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"Blender {device} render exceeded the {args.render_timeout}s "
                f"per-object timeout; see {log_path}",
            )
        if completed.returncode != 0:
            return (
                False,
                f"Blender {device} render exited with {completed.returncode}; "
                f"see {log_path}",
            )
        if not render_manifest_matches(
            cache_dir,
            sample_dir,
            camera_json,
            args,
            alignment,
            resolution,
            len(frames),
            render_visuals,
            render_device=device,
        ):
            return (
                False,
                f"Blender {device} render did not create a complete cache; "
                f"see {log_path}",
            )
        return True, None

    requested_log = cache_dir / (
        "blender_gpu.log" if args.render_device == "GPU" else "blender_cpu.log"
    )
    success, reason = run_blender(
        args.render_device, requested_log, force=args.force_rerender
    )
    if success:
        return True, None

    should_fallback = (
        args.render_engine == "CYCLES"
        and args.render_device == "GPU"
        and not args.no_render_cpu_fallback
    )
    if not should_fallback:
        return False, reason

    fallback_log = cache_dir / "blender_cpu_fallback.log"
    print(
        f"  GPU rendering failed for {sample_dir.name}; retrying with Cycles CPU "
        f"(GPU log: {requested_log})",
        flush=True,
    )
    fallback_success, fallback_reason = run_blender("CPU", fallback_log, force=True)
    if fallback_success:
        return True, None
    return (
        False,
        f"GPU render failed ({reason}); CPU fallback also failed "
        f"({fallback_reason})",
    )


def appearance_metrics(gt_render_dir: Path, pred_render_dir: Path) -> dict[str, Any]:
    transforms = load_json(gt_render_dir / "transforms.json")
    full_scores = []
    foreground_scores = []
    per_view = []
    for index, frame in enumerate(transforms.get("frames", [])):
        gt_path = gt_render_dir / frame["file_path"]
        pred_path = pred_render_dir / f"{index:03d}.png"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt_rgb, gt_alpha = read_composited_image(gt_path)
        pred_rgb, pred_alpha = read_composited_image(pred_path)
        if gt_rgb.shape != pred_rgb.shape:
            pred_image = (
                Image.open(pred_path)
                .convert("RGBA")
                .resize((gt_rgb.shape[1], gt_rgb.shape[0]), Image.Resampling.LANCZOS)
            )
            rgba = np.asarray(pred_image, dtype=np.float64) / 255.0
            alpha = rgba[..., 3:4]
            pred_rgb = rgba[..., :3] * alpha + (1.0 - alpha)
            pred_alpha = alpha[..., 0]
        union = (gt_alpha > 0.01) | (pred_alpha > 0.01)
        full = psnr_value(gt_rgb, pred_rgb)
        foreground = psnr_value(gt_rgb, pred_rgb, union)
        full_scores.append(full)
        foreground_scores.append(foreground)
        per_view.append(
            {"view": index, "psnr_db": full, "foreground_psnr_db": foreground}
        )
    if not full_scores:
        return {"available": False, "reason": "no matching GT/prediction render pairs"}
    return {
        "available": True,
        "psnr_db": float(np.mean(full_scores)),
        "psnr_std_db": float(np.std(full_scores)),
        "psnr_foreground_db": float(np.mean(foreground_scores)),
        "psnr_foreground_std_db": float(np.std(foreground_scores)),
        "num_render_views": len(full_scores),
        "psnr_background": "RGBA images composited over white",
        "per_view_psnr": per_view,
    }


def executability_metrics(urdf_path: Path | None, steps: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": True,
        "urdf_xml_parse_success": False,
        "pybullet_load_success": False,
        "pybullet_step_success": False,
        "physics_engine_executable": 0.0,
        "engine": "PyBullet DIRECT (isolated subprocess)",
    }
    if urdf_path is None or not urdf_path.exists():
        result["available"] = False
        result["reason"] = f"URDF does not exist: {urdf_path}"
        return result
    try:
        ET.parse(urdf_path)
        result["urdf_xml_parse_success"] = True
    except Exception as exc:
        result["urdf_error"] = str(exc)
        return result
    worker = WORKER_DIR / "check_urdf_executability.py"
    command = [
        sys.executable,
        str(worker),
        "--urdf",
        str(urdf_path.resolve()),
        "--steps",
        str(max(1, steps)),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        result["engine_error"] = "PyBullet executability worker timed out"
        return result
    marker = "EVALUATION_RESULT="
    marker_index = completed.stdout.rfind(marker)
    if completed.returncode != 0 or marker_index < 0:
        result["worker_returncode"] = completed.returncode
        result["engine_error"] = (
            "PyBullet worker crashed or returned no result; "
            + completed.stdout[-2000:]
        )
        return result
    try:
        worker_result = json.loads(
            completed.stdout[marker_index + len(marker) :].splitlines()[0]
        )
        result.update(worker_result)
    except Exception as exc:
        result["engine_error"] = f"unable to parse PyBullet worker result: {exc}"
    return result


def unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def evaluate_sample(spec: SampleSpec, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction_root": str(spec.root),
        "prediction_set": spec.prediction_set,
        "prediction_method": spec.adapter,
        "sample_id": spec.sample_dir.name,
        "gt_id": spec.gt_id,
        "status": "ok",
        "warnings": [],
    }
    gt_json = args.dataset_root / "finaljson" / f"{spec.gt_id}.json"
    if not gt_json.exists():
        result["status"] = "missing_gt"
        reason = f"GT JSON does not exist: {gt_json}"
        result["warnings"].append(reason)
        for key in ("geometry", "scale", "semantics", "articulation"):
            result[key] = unavailable(reason)
        return result

    try:
        asset = adapt_prediction(spec)
    except FileNotFoundError as exc:
        result["status"] = "missing_prediction"
        reason = str(exc)
        result["warnings"].append(reason)
        for key in ("geometry", "scale", "semantics", "articulation", "executability"):
            result[key] = unavailable(reason)
        return result
    result["warnings"].extend(asset.warnings)
    result["geometry_provenance"] = asset.scale_source
    if spec.adapter == "urdf_anything":
        result["warnings"].append(
            "URDF-Anything geometry is inherited from its input mesh; geometry metrics do not isolate articulation prediction."
        )
    elif spec.adapter == "articulate_anything":
        result["warnings"].append(
            "Articulate-Anything URDF visuals reference retrieved PartNet meshes; geometry metrics include retrieval/oracle geometry."
        )

    if args.skip_executability:
        result["executability"] = unavailable("disabled by --skip-executability")
    else:
        result["executability"] = executability_metrics(
            asset.urdf_path, args.simulation_steps
        )

    gt_data = load_json(gt_json)
    pred_data = asset.pred_data
    semantic_fields = (
        {"material", "priority_rank"}
        if args.score_missing_semantics_as_zero
        else asset.semantic_fields
    )
    result["semantics"], name_part_matches = semantic_metrics(
        pred_data, gt_data, semantic_fields
    )
    result["semantics"]["native_material_output_available"] = (
        "material" in asset.semantic_fields
    )
    result["semantics"]["native_affordance_output_available"] = (
        "priority_rank" in asset.semantic_fields
    )
    result["semantics"]["unsupported_metric_policy"] = (
        "score_as_all_missing"
        if args.score_missing_semantics_as_zero
        else "report_na"
    )

    gt_paths = gt_mesh_paths(args.dataset_root, spec.gt_id, gt_data)
    if not gt_paths:
        reason = "GT mesh files are missing"
        result["warnings"].append(reason)
        result["geometry"] = unavailable(reason)
        result["articulation"] = unavailable("geometry alignment unavailable")
        return result

    seed = stable_seed(
        args.seed, spec.adapter, spec.prediction_set, spec.sample_dir.name, spec.gt_id
    )
    pred_mesh = asset.mesh
    gt_mesh = load_mesh_files(gt_paths)
    pred_norm = mesh_normalization(pred_mesh)
    gt_norm = mesh_normalization(gt_mesh)
    if spec.adapter == "physx3d" and pred_data.get("_physx3d_metadata"):
        pred_data["group_info"] = groups_from_physx3d_metadata(
            pred_data["_physx3d_metadata"]
        )
    elif spec.adapter != "covetwin":
        pred_data["group_info"] = groups_from_urdf_joints(
            asset.joints, pred_norm.scale
        )

    result["scale"] = mesh_scale_metrics(
        pred_mesh, gt_mesh, gt_data, asset.scale_source
    )
    reported_scale = scale_metrics(pred_data, gt_data)
    if reported_scale.get("available"):
        result["reported_scale"] = reported_scale

    pred_points = pred_norm.points(
        sample_surface(pred_mesh, args.surface_samples, seed)
    )
    gt_points = gt_norm.points(sample_surface(gt_mesh, args.surface_samples, seed + 1))
    alignment = estimate_alignment(pred_points, gt_points, args, seed + 2)
    result["geometry"] = geometry_metrics(
        pred_points, gt_points, alignment, args.fscore_threshold
    )
    result["geometry"]["available"] = True
    result["geometry"]["pred_mesh_files"] = len(asset.render_visuals)
    result["geometry"]["gt_mesh_files"] = len(gt_paths)
    result["geometry"]["source"] = asset.scale_source

    geometric_matches, geometric_match_details = geometric_part_matches(
        asset.part_meshes,
        gt_part_meshes(args.dataset_root, spec.gt_id, gt_data),
        pred_norm,
        gt_norm,
        alignment,
        seed + 3,
    )
    part_matches = geometric_matches or name_part_matches
    result["semantics"]["geometric_part_matches"] = geometric_match_details
    result["semantics"]["articulation_part_match_method"] = (
        "geometry" if geometric_matches else "name_or_label"
    )

    if result["scale"].get("available"):
        result["articulation"] = articulation_metrics(
            pred_data,
            gt_data,
            part_matches,
            pred_norm,
            gt_norm,
            alignment,
            result["scale"],
        )
    else:
        result["articulation"] = unavailable("metric scale unavailable")

    if args.skip_psnr:
        result["geometry"].update(
            {"psnr_available": False, "psnr_reason": "disabled by --skip-psnr"}
        )
    else:
        gt_render_dir = args.renders_root / spec.gt_id
        if not (gt_render_dir / "transforms.json").exists():
            result["geometry"].update(
                {
                    "psnr_available": False,
                    "psnr_reason": f"GT camera/render data missing: {gt_render_dir}",
                }
            )
        else:
            cache_dir = (
                args.output_dir
                / "render_cache"
                / spec.prediction_set
                / spec.sample_dir.name
            )
            ready, reason = ensure_prediction_renders(
                spec.sample_dir,
                gt_render_dir,
                cache_dir,
                alignment,
                args,
                asset.render_visuals,
            )
            if ready:
                appearance = appearance_metrics(gt_render_dir, cache_dir)
                if appearance.get("available"):
                    result["geometry"].update(appearance)
                    result["geometry"]["psnr_available"] = True
                else:
                    result["geometry"].update(
                        {
                            "psnr_available": False,
                            "psnr_reason": appearance.get("reason"),
                        }
                    )
            else:
                result["geometry"].update(
                    {"psnr_available": False, "psnr_reason": reason}
                )
                result["warnings"].append(str(reason))
    return result


def nested_value(item: dict[str, Any], path: str) -> Any:
    value: Any = item
    for component in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value


def aggregate_records(
    results: list[dict[str, Any]], prediction_set: str
) -> dict[str, Any]:
    subset = [
        item
        for item in results
        if prediction_set == "all" or item["prediction_set"] == prediction_set
    ]
    summary: dict[str, Any] = {
        "prediction_set": prediction_set,
        "num_objects": len(subset),
        "num_ok": sum(item.get("status") == "ok" for item in subset),
        "num_missing_gt": sum(item.get("status") == "missing_gt" for item in subset),
        "num_missing_prediction": sum(
            item.get("status") == "missing_prediction" for item in subset
        ),
        "num_error": sum(item.get("status") == "error" for item in subset),
        "metrics": {},
    }
    for path in SUMMARY_METRICS:
        values = []
        for item in subset:
            value = nested_value(item, path)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        summary["metrics"][path] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "count": len(values),
        }

    for semantic_name, field_name in (
        ("material", "material_records"),
        ("affordance", "affordance_records"),
    ):
        records = []
        for item in subset:
            records.extend(item.get("semantics", {}).get(field_name, []))
        macro, micro = f1_scores(records)
        summary["metrics"][f"semantics.global_{semantic_name}_macro_f1"] = {
            "mean": macro,
            "std": None,
            "count": len(records),
        }
        summary["metrics"][f"semantics.global_{semantic_name}_micro_f1"] = {
            "mean": micro,
            "std": None,
            "count": len(records),
        }

    correct = sum(
        int(item.get("articulation", {}).get("joint_type_correct", 0))
        for item in subset
    )
    denominator = sum(
        int(item.get("articulation", {}).get("joint_type_denominator", 0))
        for item in subset
    )
    summary["metrics"]["articulation.global_joint_type_accuracy"] = {
        "mean": None if denominator == 0 else float(correct / denominator),
        "std": None,
        "count": denominator,
    }
    return summary


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_results = json_safe(results)
    (args.output_dir / "per_object.json").write_text(
        json.dumps(safe_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    set_names = sorted({item["prediction_set"] for item in results})
    summaries = [aggregate_records(results, name) for name in set_names]
    summaries.append(aggregate_records(results, "all"))
    method_summaries = []
    for method in sorted({item.get("prediction_method", "unknown") for item in results}):
        subset = [item for item in results if item.get("prediction_method") == method]
        summary = aggregate_records(subset, "all")
        summary["prediction_method"] = method
        summary["prediction_set"] = "all_splits"
        method_summaries.append(summary)
    protocol = {
        "adapters": "CoVeTwin: basic_info/objs/basic.urdf; URDF-Anything: mesh_reconstruction/prediction.json; Articulate-Anything: joint_actor with link_placement fallback; PhysX-3D: texture.glb plus urdf_export.",
        "geometry": "Zero-pose visual meshes with URDF joint/visual transforms applied; independent largest-extent normalization; rigid cube+trimmed-ICP alignment; symmetric CD-L2 and F-score.",
        "fscore_threshold": args.fscore_threshold,
        "appearance": "The same transformed/textured prediction sources are rendered at all GT cameras; mean PSNR after white-background RGBA compositing.",
        "scale": "Uniform proxy: calibrate GT raw mesh units with annotated GT largest dimension, apply that centimeters-per-unit factor to prediction raw extents, then compare sorted 3D dimensions. Native reported scale is retained separately when available.",
        "material_affordance": (
            "Macro/micro F1 for discrete labels; unsupported outputs are scored as all-missing (0)."
            if args.score_missing_semantics_as_zero
            else "Macro/micro F1 for exported discrete labels; unsupported outputs remain null/N/A."
        ) + " Affordance is priority_rank as a 1-10 class; visualization videos are never reverse-engineered into labels.",
        "part_matching": "Aligned per-part surface Chamfer plus Hungarian assignment; label/name matching is the fallback.",
        "articulation": "URDF/native joint types, world-frame axes/origins and ranges after the same geometry alignment.",
        "executability": "Every URDF is loaded, joint-driven and stepped in isolated PyBullet DIRECT; worker crash, exception or non-finite state scores 0.",
        "provenance_warning": "URDF-Anything inherits its input geometry; Articulate-Anything may reference retrieved/GT PartNet meshes. These geometry/PSNR values must be reported with provenance.",
    }
    summary_payload = {
        "protocol": protocol,
        "summaries": json_safe(summaries),
        "method_summaries": json_safe(method_summaries),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_fields = [
        "prediction_method",
        "prediction_set",
        "sample_id",
        "gt_id",
        "status",
    ] + SUMMARY_METRICS
    with (args.output_dir / "per_object.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in results:
            row = {key: item.get(key) for key in csv_fields[:5]}
            row.update({path: nested_value(item, path) for path in SUMMARY_METRICS})
            writer.writerow(row)

    lines = [
        "Unified articulated-asset evaluation protocol",
        "==============================================",
        "",
        json.dumps(protocol, indent=2, ensure_ascii=False),
        "",
        "Outputs:",
        "- per_object.json: full metrics, matches, warnings and per-view PSNR",
        "- per_object.csv: compact table for paper analysis",
        "- summary.json: per-split and overall aggregates",
        "- render_cache/: aligned prediction renders and Blender logs",
        "",
        "Important: test_demo/2 maps to GT 2230, which is reported as unavailable when that GT is absent.",
    ]
    (args.output_dir / "PROTOCOL.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def print_summary(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\nEvaluation complete")
    print(f"  per-object JSON: {args.output_dir / 'per_object.json'}")
    print(f"  compact CSV:     {args.output_dir / 'per_object.csv'}")
    print(f"  summary JSON:    {args.output_dir / 'summary.json'}")
    for prediction_set in sorted({item["prediction_set"] for item in results}):
        summary = aggregate_records(results, prediction_set)
        metrics = summary["metrics"]
        print(
            f"  {prediction_set}: objects={summary['num_objects']}, ok={summary['num_ok']}, "
            f"missing_gt={summary['num_missing_gt']}, "
            f"missing_prediction={summary['num_missing_prediction']}, "
            f"error={summary['num_error']}, "
            f"CD={metrics['geometry.chamfer_l2']['mean']}, "
            f"F={metrics['geometry.fscore']['mean']}, "
            f"PSNR={metrics['geometry.psnr_db']['mean']}"
        )


def main() -> int:
    args = parse_args()
    args.pred_roots = [path.resolve() for path in args.pred_roots]
    args.urdf_anything_roots = [
        path.resolve() for path in args.urdf_anything_roots
    ]
    args.articulate_roots = [path.resolve() for path in args.articulate_roots]
    args.physx3d_roots = [path.resolve() for path in args.physx3d_roots]
    args.dataset_root = args.dataset_root.resolve()
    args.renders_root = args.renders_root.resolve()
    args.output_dir = args.output_dir.resolve()
    mapping = dict(DEFAULT_TEST_DEMO_MAP)
    if args.mapping_json:
        mapping.update(
            {
                str(key): str(value)
                for key, value in load_json(args.mapping_json).items()
            }
        )

    samples = discover_samples(args, mapping)
    if not samples:
        print("No prediction samples matched the arguments.", file=sys.stderr)
        return 2
    print(
        f"Discovered {len(samples)} objects across {len({spec.root for spec in samples})} prediction roots."
    )
    results = []
    for index, spec in enumerate(samples, start=1):
        print(
            f"[{index:02d}/{len(samples):02d}] {spec.prediction_set}/{spec.sample_dir.name} "
            f"({spec.adapter}) -> GT {spec.gt_id}",
            flush=True,
        )
        try:
            result = evaluate_sample(spec, args)
        except Exception as exc:
            if args.fail_fast:
                raise
            result = {
                "prediction_root": str(spec.root),
                "prediction_set": spec.prediction_set,
                "prediction_method": spec.adapter,
                "sample_id": spec.sample_dir.name,
                "gt_id": spec.gt_id,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "warnings": [str(exc)],
            }
            print(f"  [ERROR] {exc}", file=sys.stderr)
        results.append(result)
        write_outputs(results, args)
    print_summary(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
