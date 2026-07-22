"""Legacy v2 voxel-token codec retained for data and ablation compatibility."""

import re
from collections import deque

import numpy as np


VOXEL_CODEC_PROMPT = (
    "Return the part occupancy using compact PhysX voxel codec v2. "
    "Use either `v2b x,y,z,dx,dy,dz ...` for axis-aligned occupied boxes "
    "or `v2r 184 198 199-216 ...` for run-length indices. "
    "Grid is 32 and coordinates/counts are integers."
)

VOXEL_COT_PROMPT = (
    VOXEL_CODEC_PROMPT
    + " First reason about the part support, scale, and connectivity from the "
    "image and structured part description, then output only the final voxel "
    "codec string. Do not output geometry hints, joint hints, or scaffold tags."
)


def voxel_encode_indices(voxels: np.ndarray, size: int = 32) -> np.ndarray:
    voxels = _normalize_voxels(voxels, size)
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    return (x << 10) | (y << 5) | z


def voxel_decode_indices(indices: np.ndarray, size: int = 32) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).ravel()
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    indices = np.clip(indices, 0, size**3 - 1)
    x = (indices >> 10) & 31
    y = (indices >> 5) & 31
    z = indices & 31
    return np.stack([x, y, z], axis=1).astype(np.int64)


def ints_to_run_string(indices: np.ndarray) -> str:
    nums = sorted(set(np.asarray(indices, dtype=np.int64).ravel().tolist()))
    if not nums:
        return ""
    result = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        result.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = n
    result.append(f"{start}-{prev}" if start != prev else str(start))
    return " ".join(result)


def run_string_to_ints(text: str) -> np.ndarray:
    text = _strip_known_prefix(text)
    if not text.strip():
        return np.array([], dtype=np.int64)
    out = []
    for token in re.findall(r"\d+\s*-\s*\d+|\d+", text):
        if "-" in token:
            a, b = [int(x) for x in re.findall(r"\d+", token)[:2]]
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    return np.array(sorted(set(out)), dtype=np.int64)


def encode_voxels(voxels: np.ndarray, size: int = 32, mode: str = "auto") -> str:
    voxels = _normalize_voxels(voxels, size)
    rle_body = ints_to_run_string(voxel_encode_indices(voxels, size))
    rle = "v2r " + rle_body if rle_body else "v2r"

    boxes = greedy_box_decompose(voxels, size)
    box_body = " ".join(",".join(map(str, box)) for box in boxes)
    box = "v2b " + box_body if box_body else "v2b"

    if mode == "rle":
        return rle
    if mode == "box":
        return box
    if mode != "auto":
        raise ValueError(f"Unknown voxel codec mode: {mode}")
    return box if len(box) < len(rle) else rle


def decode_voxel_string(text: str, size: int = 32) -> np.ndarray:
    text = strip_scaffold(text)[1].strip()
    if text.startswith("v2b"):
        return boxes_to_voxels(_parse_boxes(text[3:]), size)
    if text.startswith("v2r"):
        return voxel_decode_indices(run_string_to_ints(text[3:]), size)
    return voxel_decode_indices(run_string_to_ints(text), size)


def strip_scaffold(text: str) -> tuple[str, str]:
    text = text.strip()
    scaffold = ""
    match = re.search(r"<scaffold>(.*?)</scaffold>", text, flags=re.DOTALL)
    if match:
        scaffold = match.group(0).strip()
        text = text[match.end():].strip()
    payload = re.search(r"\bv2[br]\b", text)
    if payload:
        text = text[payload.start():].strip()
    return scaffold, text


def codec_stats(voxels: np.ndarray, size: int = 32) -> dict:
    voxels = _normalize_voxels(voxels, size)
    legacy = ints_to_run_string(voxel_encode_indices(voxels, size))
    compact = encode_voxels(voxels, size=size, mode="auto")
    boxes = greedy_box_decompose(voxels, size)
    return {
        "n_voxels": int(len(voxels)),
        "n_boxes": int(len(boxes)),
        "legacy_chars": int(len(legacy)),
        "compact_chars": int(len(compact)),
        "char_ratio": float(len(compact) / max(1, len(legacy))),
        "mode": compact.split(" ", 1)[0],
    }


def geometry_teacher_hint(voxels: np.ndarray, size: int = 32, n_anchor: int = 8) -> str:
    voxels = _normalize_voxels(voxels, size)
    if len(voxels) == 0:
        return "hint n=0 bbox=none anchor=none boundary=0"

    bmin = voxels.min(axis=0)
    bmax = voxels.max(axis=0)
    anchors = _farthest_point_anchors(voxels, n_anchor)
    boundary_count = int(_boundary_mask(voxels, size).sum())
    anchor_text = " ".join(",".join(map(str, p.tolist())) for p in anchors)
    return (
        f"hint n={len(voxels)} "
        f"bbox={','.join(map(str, bmin.tolist()))}-{','.join(map(str, bmax.tolist()))} "
        f"anchor={anchor_text} boundary={boundary_count}"
    )


def physics_teacher_hint(jsondata: dict, part_idx: int) -> str:
    group_info = jsondata.get("group_info", {})
    matches = []
    for group_id, group in group_info.items():
        if group_id == "0" or not isinstance(group, list) or len(group) < 4:
            continue
        child_parts, parent_group, params, joint_type = group[:4]
        if part_idx not in child_parts:
            continue
        if joint_type == "B":
            axis = _round_list(params[:3])
            limit = _round_list(params[6:8])
            matches.append(f"group={group_id} parent={parent_group} joint=slide axis={axis} limit={limit}")
        elif joint_type == "C":
            axis = _round_list(params[:3])
            origin = _round_list(params[3:6])
            limit = _round_list(params[6:8])
            matches.append(f"group={group_id} parent={parent_group} joint=hinge axis={axis} origin={origin} limit={limit}")
        elif joint_type == "D":
            origin = _round_list(params[3:6])
            matches.append(f"group={group_id} parent={parent_group} joint=ball origin={origin}")
        elif joint_type == "CB":
            axis = _round_list(params[:3])
            origin = _round_list(params[3:6])
            slide_axis = _round_list(params[8:11])
            matches.append(
                f"group={group_id} parent={parent_group} joint=hinge_slide "
                f"axis={axis} origin={origin} slide_axis={slide_axis}"
            )
        else:
            matches.append(f"group={group_id} parent={parent_group} joint={joint_type}")
    if not matches:
        return "physics root_or_fixed: keep support volume coherent and avoid isolated voxel islands"
    return "physics " + " | ".join(matches)


def scaffold_text(
    voxels: np.ndarray,
    jsondata: dict,
    part_idx: int,
    use_geometry: bool = True,
    use_physics: bool = True,
) -> str:
    fields = []
    if use_geometry:
        fields.append(geometry_teacher_hint(voxels))
    if use_physics:
        fields.append(physics_teacher_hint(jsondata, part_idx))
    if not fields:
        fields.append("hint infer compact geometry support and joint-consistent occupancy")
    return "<scaffold> " + " ; ".join(fields) + " </scaffold>"


def cot_voxel_answer(
    voxels: np.ndarray,
    jsondata: dict,
    part_idx: int,
    mode: str = "auto",
    use_geometry: bool = True,
    use_physics: bool = True,
) -> str:
    return encode_voxels(voxels, mode=mode)


def voxel_rerank_score(voxels: np.ndarray, size: int = 32) -> tuple[float, dict]:
    quality = validate_decoded_voxels(voxels, size)
    n_voxels = quality["n_voxels"]
    if n_voxels <= 0:
        return -1e9, quality
    largest = quality["largest_component_ratio"]
    components = quality["n_components"]
    score = 100.0 * largest - 2.0 * components + min(n_voxels, 32768) / 32768.0
    return float(score), quality


def validate_decoded_voxels(voxels: np.ndarray, size: int = 32) -> dict:
    voxels = _normalize_voxels(voxels, size)
    if len(voxels) == 0:
        return {"n_voxels": 0, "n_components": 0, "largest_component_ratio": 0.0}
    n_components, largest = _component_stats(voxels, size)
    return {
        "n_voxels": int(len(voxels)),
        "n_components": int(n_components),
        "largest_component_ratio": float(largest / max(1, len(voxels))),
    }


def _normalize_voxels(voxels: np.ndarray, size: int) -> np.ndarray:
    voxels = np.asarray(voxels, dtype=np.int64)
    if voxels.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError("voxels must have shape (N, 3)")
    voxels = np.clip(voxels, 0, size - 1)
    voxels = np.unique(voxels, axis=0)
    order = np.lexsort((voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    return voxels[order]


def greedy_box_decompose(voxels: np.ndarray, size: int = 32) -> list[tuple[int, int, int, int, int, int]]:
    occ = np.zeros((size, size, size), dtype=bool)
    voxels = _normalize_voxels(voxels, size)
    occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    boxes = []
    while occ.any():
        x, y, z = np.argwhere(occ)[0].tolist()
        dz = 1
        while z + dz < size and occ[x, y, z + dz]:
            dz += 1
        dy = 1
        while y + dy < size and occ[x, y + dy, z:z + dz].all():
            dy += 1
        dx = 1
        while x + dx < size and occ[x + dx, y:y + dy, z:z + dz].all():
            dx += 1
        occ[x:x + dx, y:y + dy, z:z + dz] = False
        boxes.append((x, y, z, dx, dy, dz))
    return boxes


def boxes_to_voxels(boxes: list[tuple[int, int, int, int, int, int]], size: int = 32) -> np.ndarray:
    coords = []
    for x, y, z, dx, dy, dz in boxes:
        x0, y0, z0 = np.clip([x, y, z], 0, size - 1)
        x1 = int(np.clip(x0 + max(1, dx), 0, size))
        y1 = int(np.clip(y0 + max(1, dy), 0, size))
        z1 = int(np.clip(z0 + max(1, dz), 0, size))
        for xi in range(int(x0), x1):
            for yi in range(int(y0), y1):
                for zi in range(int(z0), z1):
                    coords.append((xi, yi, zi))
    return _normalize_voxels(np.array(coords, dtype=np.int64), size)


def _parse_boxes(text: str) -> list[tuple[int, int, int, int, int, int]]:
    boxes = []
    for token in text.split():
        nums = [int(x) for x in re.findall(r"-?\d+", token)]
        if len(nums) >= 6:
            boxes.append(tuple(nums[:6]))
    return boxes


def _strip_known_prefix(text: str) -> str:
    text = text.strip()
    if text.startswith("v2r") or text.startswith("v2b"):
        return text[3:].strip()
    return text


def _farthest_point_anchors(voxels: np.ndarray, n_anchor: int) -> np.ndarray:
    if len(voxels) <= n_anchor:
        return voxels
    center = voxels.mean(axis=0, keepdims=True)
    first = int(np.linalg.norm(voxels - center, axis=1).argmax())
    chosen = [first]
    min_dist = np.linalg.norm(voxels - voxels[first], axis=1)
    for _ in range(1, n_anchor):
        idx = int(min_dist.argmax())
        chosen.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(voxels - voxels[idx], axis=1))
    anchors = voxels[chosen]
    order = np.lexsort((anchors[:, 2], anchors[:, 1], anchors[:, 0]))
    return anchors[order]


def _boundary_mask(voxels: np.ndarray, size: int) -> np.ndarray:
    occ = np.zeros((size, size, size), dtype=bool)
    occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    mask = np.zeros(len(voxels), dtype=bool)
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for i, v in enumerate(voxels):
        neigh = v[None, :] + offsets
        outside = (neigh < 0).any(axis=1) | (neigh >= size).any(axis=1)
        inside = neigh[~outside]
        if outside.any() or not occ[inside[:, 0], inside[:, 1], inside[:, 2]].all():
            mask[i] = True
    return mask


def _component_stats(voxels: np.ndarray, size: int) -> tuple[int, int]:
    occ = np.zeros((size, size, size), dtype=bool)
    occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    seen = np.zeros_like(occ)
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    n_components = 0
    largest = 0
    for start in map(tuple, voxels.tolist()):
        if seen[start]:
            continue
        n_components += 1
        count = 0
        queue = deque([start])
        seen[start] = True
        while queue:
            x, y, z = queue.popleft()
            count += 1
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size and occ[nx, ny, nz] and not seen[nx, ny, nz]:
                    seen[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
        largest = max(largest, count)
    return n_components, largest


def _round_list(values, ndigits: int = 3) -> str:
    return "[" + ",".join(str(round(float(v), ndigits)) for v in values) + "]"
