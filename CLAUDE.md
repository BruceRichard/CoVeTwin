# CLAUDE.md

This file provides repository-specific guidance for coding agents working on
CoVeTwin.

## Project scope

CoVeTwin generates a high-fidelity articulated digital twin from one image.
Its four inference stages are:

1. compact part-geometry reasoning and multi-candidate verification;
2. image-conditioned high-resolution flow reconstruction;
3. coarse-to-fine part-label transfer;
4. physical metadata, URDF, and MuJoCo XML export.

The public launcher is `run_covetwin.py`. The numbered backend entry points are
kept under `pipeline/`, while reusable method code lives under `covetwin/`.

## Important paths

- `covetwin/geometry_codec.py`: relative shape-span representation.
- `covetwin/verification.py`: validity checks and candidate score.
- `covetwin/inference.py`: two-turn VLM inference implementation.
- `covetwin/flow_matching.py`: paper-aligned flow objective helpers.
- `pipeline/1_geometry_reasoning.py`: stage-1 CLI.
- `pipeline/2_flow_reconstruction.py`: TRELLIS-based stage-2 decoder.
- `pipeline/3_part_segmentation.py`: refined mesh segmentation.
- `pipeline/4_simulation_export.py`: URDF and MJCF construction.
- `training/build_dataset.py`: CoVeTwin fine-tuning records.
- `evaluation/evaluate_metrics.py`: unified quantitative evaluation.

## Contracts to preserve

Stage 1 writes `basic_info.txt`, `coord_<part>.txt`, `ind_<part>.npy`, and
`allind.npy`. It also writes `candidate_verification.json` and per-candidate
geometry strings. Stages 2--4 consume this layout and add `sample.glb`,
`objs/`, `basic_info.json`, `basic.urdf`, and `basic.xml`.

The canonical geometry text uses relative shape spans:

```text
rss <base> <relative_start>:<length> ...
```

Do not silently accept malformed, empty, overlapping, unordered, or
out-of-range spans. Candidate verification uses 6-connectivity and the exact
paper score implemented in `covetwin/verification.py`.

## Common commands

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/download_checkpoints.py

python run_covetwin.py \
  --demo-path demo \
  --output-path test_covetwin \
  --ckpt pretrain/covetwin_vlm \
  --candidate-count 5

python training/build_dataset.py \
  --representation relative_span \
  --output dataset/covetwin_training/conversations.json

python -m unittest discover -s tests -p 'test_covetwin*.py' -v
```

## Environment notes

- Use Python 3.10 and the project conda environment.
- The high-resolution decoder requires CUDA and compiled sparse/rendering
  extensions.
- Prefer the bundled Blender 3.6 executable for evaluation rendering. The
  system Blender 2.82 cannot compile Cycles kernels for RTX 4090 GPUs.
- Set `HF_ENDPOINT=https://hf-mirror.com` for Hugging Face downloads.
- Generated assets, datasets, checkpoints, videos, and evaluation results are
  intentionally excluded from Git by `.gitignore`.
