#!/usr/bin/env python3
"""Match wild-demo images to semantically compatible rendered objects."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


def normalize_name(value: object) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())


def category_predicate(stem: str) -> Callable[[str, str], bool]:
    predicates: dict[str, Callable[[str, str], bool]] = {
        "bin": lambda name, category: name in {"trashcan", "trash can"},
        "chair": lambda name, category: "chair" in name,
        "drawer": lambda name, category: (
            "storage furniture" in category
            and any(word in name for word in ("cabinet", "wardrobe", "dresser", "drawer"))
        ),
        "lab": lambda name, category: "lamp" in name,
        "microwave": lambda name, category: name == "microwave" or "oven" in name,
        "scissors": lambda name, category: "scissor" in name or "shear" in name,
        "storage": lambda name, category: (
            "storage furniture" in category
            and any(
                word in name
                for word in ("cabinet", "drawer", "dresser", "sideboard", "storage table")
            )
        ),
        "switch": lambda name, category: (
            "switch" in name and "electrical control" in category
        ),
        "toilet": lambda name, category: name == "toilet",
        "windows": lambda name, category: "window" in name,
    }
    if stem not in predicates:
        raise KeyError(f"No semantic category rule for query {stem!r}")
    return predicates[stem]


def square_on_white(image: Image.Image, crop_alpha: bool) -> Image.Image:
    image.load()
    if image.mode == "RGBA":
        if crop_alpha:
            bbox = image.getchannel("A").getbbox()
            if bbox:
                image = image.crop(bbox)
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    side = max(image.size)
    margin = max(4, int(side * 0.06))
    canvas = Image.new("RGB", (side + 2 * margin, side + 2 * margin), "white")
    canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
    return canvas


class Encoder:
    def __init__(self, repo: Path, device: str, batch_size: int) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = torch.hub.load(
            str(repo), "dinov2_vitb14_reg", source="local", pretrained=True
        ).eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @torch.inference_mode()
    def encode(self, paths: list[Path], crop_alpha: bool) -> torch.Tensor:
        outputs = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = torch.stack(
                [
                    self.transform(square_on_white(Image.open(path), crop_alpha))
                    for path in batch_paths
                ]
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                features = self.model(images)
            outputs.append(F.normalize(features.float(), dim=-1).cpu())
            print(f"    encoded {min(start + len(batch_paths), len(paths))}/{len(paths)}")
        return torch.cat(outputs, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", type=Path, default=Path("wild-demo"))
    parser.add_argument(
        "--renders-root", type=Path, default=Path("dataset_toolkits/renders_all")
    )
    parser.add_argument(
        "--metadata-root", type=Path, default=Path("dataset/PhysX_mobility/finaljson")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("wild-demo-new"))
    parser.add_argument(
        "--torch-hub-repo",
        type=Path,
        default=Path("/home/ices204/.cache/torch/hub/facebookresearch_dinov2_main"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--view",
        default="000",
        help="Single zero-padded rendered view to compare, for example 000.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    queries = sorted(
        path
        for path in args.query_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    render_ids = {path.name for path in args.renders_root.iterdir() if path.is_dir()}
    metadata: dict[str, dict[str, str]] = {}
    for path in args.metadata_root.glob("*.json"):
        if path.stem not in render_ids:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata[path.stem] = {
            "object_name": str(data.get("object_name", "")),
            "category": str(data.get("category", "")),
        }

    candidate_ids: dict[str, list[str]] = {}
    for query in queries:
        predicate = category_predicate(query.stem)
        ids = [
            object_id
            for object_id, info in metadata.items()
            if predicate(
                normalize_name(info["object_name"]), normalize_name(info["category"])
            )
        ]
        if not ids:
            raise RuntimeError(f"No candidates for {query.name}")
        candidate_ids[query.stem] = sorted(ids, key=lambda item: int(item))
        print(f"{query.name}: {len(ids)} semantic candidate objects")

    encoder = Encoder(args.torch_hub_repo, args.device, args.batch_size)
    print("Encoding wild queries")
    query_features = encoder.encode(queries, crop_alpha=False)

    rankings: dict[str, list[dict[str, object]]] = {}
    feature_cache: dict[tuple[str, ...], tuple[list[Path], torch.Tensor]] = {}
    for query_index, query in enumerate(queries):
        ids = candidate_ids[query.stem]
        cache_key = tuple(ids)
        if cache_key not in feature_cache:
            paths = [
                args.renders_root / object_id / f"{args.view}.png"
                for object_id in ids
                if (args.renders_root / object_id / f"{args.view}.png").is_file()
            ]
            print(f"Encoding {query.stem}: {len(paths)} rendered views")
            feature_cache[cache_key] = (paths, encoder.encode(paths, crop_alpha=True))
        paths, features = feature_cache[cache_key]
        similarities = features @ query_features[query_index]
        best_by_object: dict[str, tuple[float, Path]] = {}
        for path, similarity in zip(paths, similarities.tolist()):
            object_id = path.parent.name
            current = best_by_object.get(object_id)
            if current is None or similarity > current[0]:
                best_by_object[object_id] = (float(similarity), path)
        ranking = []
        for object_id, (score, path) in sorted(
            best_by_object.items(), key=lambda item: item[1][0], reverse=True
        ):
            ranking.append(
                {
                    "object_id": object_id,
                    "view": path.name,
                    "source_image": str(path),
                    "score": score,
                    **metadata[object_id],
                }
            )
        rankings[query.stem] = ranking

    used_ids: set[str] = set()
    mapping: dict[str, dict[str, object]] = {}
    for query in queries:
        ranking = rankings[query.stem]
        selected = next(item for item in ranking if str(item["object_id"]) not in used_ids)
        used_ids.add(str(selected["object_id"]))
        destination = args.output_dir / f"{query.stem}.png"
        shutil.copy2(str(selected["source_image"]), destination)
        mapping[query.name] = {
            "output_image": str(destination),
            "selected": selected,
            "top5": ranking[:5],
        }
        print(
            f"{query.name} -> {selected['object_id']}/{selected['view']} "
            f"({selected['object_name']}, score={selected['score']:.4f})"
        )

    (args.output_dir / "similarity_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
