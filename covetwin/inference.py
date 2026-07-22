#!/usr/bin/env python3
"""CoVeTwin geometry reasoning with compact candidates and verification.

Outputs retain the file contract consumed by stages 2--4: ``basic_info.txt``,
``ind_i.npy``, and ``allind.npy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

import numpy as np
from PIL import Image

from covetwin.geometry_codec import (
    encode_relative_shape_spans,
    serialize_relative_shape_spans,
)
from covetwin.prompts import part_geometry_prompt
from covetwin.verification import CandidateSelection, evaluate_candidate, select_best_candidate


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "CoVeTwin two-stage VLM inference using relative shape spans and "
            "structure-verified geometry candidates."
        ),
    )
    parser.add_argument("--demo-path", "--demo_path", type=Path, default=Path("demo"))
    parser.add_argument("--output-path", "--output_path", type=Path, default=Path("test_covetwin"))
    parser.add_argument("--ckpt", type=str, default="./pretrain/vlm")
    parser.add_argument(
        "--processor-ckpt",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Processor source; downloads through HF_ENDPOINT when not cached.",
    )
    parser.add_argument("--global-prompt", type=Path, default=Path("dataset/overall_prompt.txt"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("-K", "--candidate-count", type=int, default=5)
    parser.add_argument(
        "--verify-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank K candidates with Eq. (16); disable for the paper's no-verification ablation.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--global-max-new-tokens", type=int, default=4096)
    parser.add_argument("--geometry-max-new-tokens", type=int, default=8192)
    parser.add_argument("--min-pixels", type=int, default=65_536)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--remove-bg", "--remove_bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-part-ply", "--save_part_ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if args.candidate_count <= 0:
        parser.error("--candidate-count must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0,1]")
    if args.candidate_count > 1 and args.temperature <= 0:
        parser.error("--temperature must be positive when sampling multiple candidates")
    return args


def _input_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _generate_text(
    model,
    processor,
    process_vision_info,
    messages: list[dict],
    *,
    sample: bool,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_input_device(model))
    generation = {
        "do_sample": sample,
        "max_new_tokens": max_new_tokens,
    }
    if sample:
        generation.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation)
    trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _part_indices(global_output: str) -> list[int]:
    indices = sorted({int(value) for value in re.findall(r"\bl_(\d+)\b", global_output)})
    if not indices:
        raise ValueError("the global response contains no l_i part labels")
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"part labels must be contiguous from l_0; found {indices}")
    return indices


def _stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little") % 1_000_000


def _save_point_cloud(path: Path, voxels: np.ndarray) -> None:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("--save-part-ply requires trimesh") from error
    trimesh.points.PointCloud(voxels).export(path)


def _candidate_messages(base_messages: list[dict], global_output: str, prompt: str) -> list[dict]:
    return base_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": global_output}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def process_image(
    image_path: Path,
    args: argparse.Namespace,
    model,
    processor,
    process_vision_info,
    global_prompt: str,
) -> dict:
    sample_id = image_path.stem
    output_dir = args.output_path / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "allind.npy").exists() and not args.force:
        print(f"[CoVeTwin] skip existing {sample_id}", flush=True)
        return {"sample_id": sample_id, "status": "skipped_existing"}
    if args.force:
        # Remove only files owned by CoVeTwin stage 1. Downstream GLB/URDF/XML
        # assets are intentionally preserved and handled by their own stages.
        for pattern in ("coord_*.txt", "ind_*.npy", "ind_*.ply"):
            for old_path in output_dir.glob(pattern):
                old_path.unlink()
        for old_name in ("allind.npy", "candidate_verification.json"):
            old_path = output_dir / old_name
            if old_path.exists():
                old_path.unlink()
        candidate_root = output_dir / "candidates"
        if candidate_root.is_dir():
            shutil.rmtree(candidate_root)
        if (output_dir / "sample.glb").exists():
            print(
                f"[CoVeTwin] warning: {sample_id}/sample.glb is a previous stage-2 "
                "asset and will not be overwritten by stage 1",
                file=sys.stderr,
                flush=True,
            )

    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    if args.remove_bg:
        try:
            from rembg import remove
        except ImportError as error:
            raise RuntimeError("--remove-bg requires the rembg package") from error
        image = remove(image).convert("RGB")

    base_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": global_prompt},
            ],
        }
    ]
    sample_seed = _stable_sample_seed(args.seed, sample_id)
    global_output = _generate_text(
        model,
        processor,
        process_vision_info,
        base_messages,
        sample=False,
        seed=sample_seed,
        max_new_tokens=args.global_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    (output_dir / "basic_info.txt").write_text(global_output, encoding="utf-8")
    part_indices = _part_indices(global_output)

    merged: list[np.ndarray] = []
    report: dict = {
        "method": "CoVeTwin",
        "sample_id": sample_id,
        "image": str(image_path.resolve()),
        "grid_size": args.grid_size,
        "candidate_count": args.candidate_count,
        "verification_enabled": args.verify_candidates,
        "score": "100*rho - 2*c + min(n,R^3)/R^3",
        "parts": [],
    }
    for part_index in part_indices:
        prompt = part_geometry_prompt(part_index, args.grid_size)
        messages = _candidate_messages(base_messages, global_output, prompt)
        candidate_dir = output_dir / "candidates" / f"part_{part_index:03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[str] = []
        for candidate_index in range(args.candidate_count):
            text = _generate_text(
                model,
                processor,
                process_vision_info,
                messages,
                sample=args.candidate_count > 1,
                seed=sample_seed + 10_000 * (part_index + 1) + candidate_index,
                max_new_tokens=args.geometry_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            candidates.append(text)
            (candidate_dir / f"candidate_{candidate_index:03d}.txt").write_text(text, encoding="utf-8")

        if args.verify_candidates:
            selection = select_best_candidate(candidates, args.grid_size)
        else:
            evaluations = []
            decoded_first = None
            for candidate_index, candidate in enumerate(candidates):
                evaluation, decoded = evaluate_candidate(
                    candidate, candidate_index, args.grid_size
                )
                evaluations.append(evaluation)
                if candidate_index == 0:
                    decoded_first = decoded
            if decoded_first is None:
                raise ValueError(
                    "the first candidate is unparsable/empty in no-verification mode"
                )
            selection = CandidateSelection(
                selected_index=0,
                voxels=decoded_first,
                evaluations=tuple(evaluations),
            )
        canonical = serialize_relative_shape_spans(
            encode_relative_shape_spans(selection.voxels, args.grid_size)
        )
        (output_dir / f"coord_{part_index}.txt").write_text(canonical, encoding="utf-8")
        np.save(output_dir / f"ind_{part_index}.npy", selection.voxels)
        if args.save_part_ply:
            _save_point_cloud(output_dir / f"ind_{part_index}.ply", selection.voxels)
        merged.append(selection.voxels)

        part_report = selection.to_dict()
        for item in part_report["candidates"]:
            item.pop("raw_text", None)
            item["file"] = str(
                (candidate_dir / f"candidate_{item['index']:03d}.txt").relative_to(
                    output_dir
                )
            )
        part_report.update(
            part_index=part_index,
            selected_codec=canonical,
            selected_voxel_file=f"ind_{part_index}.npy",
        )
        report["parts"].append(part_report)
        print(
            f"[CoVeTwin] {sample_id} l_{part_index}: selected "
            f"{selection.selected_index}/{args.candidate_count - 1}, "
            f"Q={selection.selected.score:.6f}, "
            f"verified={args.verify_candidates}",
            flush=True,
        )

    np.save(output_dir / "allind.npy", np.concatenate(merged, axis=0))
    (output_dir / "candidate_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"sample_id": sample_id, "status": "ok", "parts": len(part_indices)}


def main() -> int:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if not args.demo_path.is_dir():
        raise FileNotFoundError(args.demo_path)
    if not args.global_prompt.is_file():
        raise FileNotFoundError(args.global_prompt)
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Heavy imports occur only for actual inference, keeping --help and the
    # codec/test path lightweight.
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor_ckpt,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    global_prompt = args.global_prompt.read_text(encoding="utf-8")
    selected = set(args.only)
    images = sorted(
        path
        for path in args.demo_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not selected or path.stem in selected)
    )
    if not images:
        raise RuntimeError(f"no selected images found in {args.demo_path}")

    results = []
    for image_path in images:
        try:
            results.append(
                process_image(
                    image_path,
                    args,
                    model,
                    processor,
                    process_vision_info,
                    global_prompt,
                )
            )
        except Exception as error:  # keep a long batch useful while reporting failures
            failure = {
                "sample_id": image_path.stem,
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            results.append(failure)
            print(f"[CoVeTwin] failed {image_path.stem}: {error}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
    manifest = {
        "method": "CoVeTwin",
        "input_root": str(args.demo_path.resolve()),
        "output_root": str(args.output_path.resolve()),
        "results": results,
    }
    (args.output_path / "covetwin_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures = sum(item["status"] == "failed" for item in results)
    print(f"[CoVeTwin] complete: {len(results) - failures}/{len(results)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
