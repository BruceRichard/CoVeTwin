# CoVeTwin metric evaluation

The user-facing entry point is `evaluate_covetwin_metrics.py` in the repository
root. It supports both prediction layouts in one run:

- `test_demo/<index>` uses the built-in index-to-GT mapping.
- `test_demo_new/<object_id>` uses `<object_id>` directly, without a mapping.

## Directory layout

- `evaluate_metrics.py` contains metric computation, method adapters,
  aggregation, and the public `main()` implementation.
- `render_pred_views.py` is intentionally separate because Blender must launch
  it with Blender's embedded Python interpreter.
- `check_urdf_executability.py` is intentionally separate so a native
  PyBullet crash is isolated and recorded as an execution failure instead of
  terminating the full benchmark.

The repository-level `evaluate_covetwin_metrics.py` is a small stable CLI
wrapper. It contains no duplicate metric implementation.

## Full evaluation

Run from the CoVeTwin repository with its conda environment active:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_covetwin_metrics.py \
  --pred-roots ./test_demo ./test_demo_new \
  --dataset-root ./dataset/PhysX_mobility \
  --renders-root ./dataset_toolkits/renders_all \
  --output-dir ./evaluation_results/covetwin_metrics \
  --render-engine CYCLES \
  --render-device GPU \
  --render-samples 128
```

Prediction renders are cached. Re-running the same command resumes from the
cache. Use `--force-rerender` only when a clean render pass is needed.

For a quick non-rendering check:

```bash
python evaluate_covetwin_metrics.py \
  --pred-roots ./test_demo ./test_demo_new \
  --output-dir ./evaluation_results/covetwin_metrics_no_psnr \
  --skip-psnr
```

For one object:

```bash
python evaluate_covetwin_metrics.py \
  --pred-roots ./test_demo ./test_demo_new \
  --only test_demo:0 \
  --output-dir ./evaluation_results/object_0 \
  --skip-psnr
```

## Metric protocol

- PSNR: mean RGB PSNR over the available 25 GT cameras; RGBA is composited on
  white. Foreground-union PSNR is also reported.
- CD: symmetric squared-L2 Chamfer distance on surface samples.
- F-score: surface precision/recall with threshold 0.05 after largest-extent
  normalization, following the PhysX-3D geometry protocol.
- Absolute scale error: Euclidean distance between the sorted three physical
  dimensions in centimeters. Largest-dimension error and relative error are
  also reported.
- Material/affordance F1: label-first part matching followed by name-based
  Hungarian matching. Material strings are lexically normalized. Affordance is
  the `priority_rank` value treated as a 1--10 class.
- Articulation: group matching uses child-part IoU. Axis angle, revolute-axis
  origin distance and motion-range vector error follow Articulate-Anything.
  Revolute and prismatic range errors are separately reported in radians and
  meters.
- Executability: the method's URDF must parse, load in isolated PyBullet
  DIRECT, accept joint commands, and step for 100 frames without an exception
  or non-finite state.

Geometry is independently normalized before rigid cube-initialized ICP. The
same prediction-to-GT frame transform is used for articulation and PSNR.
Absolute scale is therefore measured separately and cannot be hidden by ICP.

## Outputs

- `per_object.json`: complete records, matches, errors and per-view PSNR.
- `per_object.csv`: compact paper-ready table.
- `summary.json`: per-split and overall aggregates.
- `PROTOCOL.txt`: protocol snapshot saved with the run.
- `render_cache/`: predicted views, Blender logs, alignment and manifests.

At present, `test_demo/2` maps to object `2230`, but the local
`dataset/PhysX_mobility` has no GT JSON/mesh/render entry for that ID. It is
kept in the results with status `missing_gt` rather than assigned fabricated
metric values.
