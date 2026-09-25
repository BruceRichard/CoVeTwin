# CoVeTwin Reproducibility and Technical Supplement

This document records the data, training, inference, evaluation, and system
configuration used by the CoVeTwin implementation. It is intended to accompany
the main paper and to make every reported number auditable against the released
code.

> **Protocol status.** Values marked **implemented** are read directly from the
> current repository. Values marked **reported** were supplied by the authors.
> The final section records the audit status of earlier discrepancies and the
> code or flags that resolve them.

## 1. Dataset and split protocol

### 1.1 PhysX-Mobility

CoVeTwin uses **PhysX-Mobility**, a physics-annotated dataset derived from
[PartNet-Mobility](https://sapien.ucsd.edu/browse). The local release contains
2,024 articulated objects with annotations for absolute scale, material,
affordance, kinematics, and functional descriptions. Each object has 25
rendered RGBA views and camera parameters, giving 50,600 object-view images.
The images stored in `dataset_toolkits/renders_all` are 1,024 x 1,024 pixels.

PhysX-Mobility should be cited directly as follows:

```bibtex
@article{physxanything,
  title   = {PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image},
  author  = {Cao, Ziang and Hong, Fangzhou and Chen, Zhaoxi and Pan, Liang and Liu, Ziwei},
  journal = {arXiv preprint arXiv:2511.13648},
  year    = {2025}
}
```

Dataset paper: [arXiv:2511.13648](https://arxiv.org/abs/2511.13648). Dataset
metadata and the same citation are included in
`dataset/PhysX_mobility/README.md`.

### 1.2 Object-level split

The split files in `dataset/splits` are disjoint and cover all 2,024 objects:

| Split | Objects | Rendered views | Part-view SFT records |
|---|---:|---:|---:|
| Training | 1,636 | 40,900 | 284,625 |
| Validation | 0 | 0 | 0 |
| Test | 388 | 9,700 | 67,775 |
| Total | 2,024 | 50,600 | 352,400 |

The part-view record counts follow the released two-turn builder: one record is
created for every `(object, part, view)` tuple. The training split contains
11,385 annotated parts and the test split contains 2,711. No validation split
is defined, and training uses `--eval_strategy no`. Model selection is therefore
checkpoint-based rather than validation-based.

The intended split is strictly object-level: all 25 views and all parts of an
object must remain in the same split. `training/build_dataset.py` enforces
this by default: `--split train` (the default) restricts the emitted records
to the 1,636 IDs in `dataset/splits/trainingset.npy`, and `--split test`
restricts them to the 388 IDs in `dataset/splits/testset.npy`. `--split all`
hard-errors unless `--allow-test-leak` is passed explicitly, and any leak is
announced with a loud warning. The split source can be overridden with
`--split-dir` or `--split-json`, `--shuffle-seed` makes the emitted record
order deterministic, and `--make-split-json` (with `--split-seed` and
`--test-count`) deterministically regenerates a split JSON when the canonical
files are absent. The train-only conversation file is therefore produced by
the default invocation:

```bash
python training/build_dataset.py \
  --voxel-root dataset/tmp_mobility/partseg \
  --structure-root dataset/txt_rep_32_finetune_mobility_all \
  --image-root dataset_toolkits/renders_all \
  --representation relative_span \
  --views-per-object 25 \
  --output dataset/covetwin_training/conversations_train.json
```

### 1.3 Evaluation-set size

Two evaluation regimes must not be conflated:

- The official object-level test split contains **388 objects**.
- The currently generated paper/demo results use 37 requested objects from
  `test_demo` and `test_demo_new`. Object `test_demo/2` maps to ID 2230, for
  which the local PhysX-Mobility GT is absent, leaving **36 commonly evaluable
  objects**.

The 36-object set is not a held-out subset under the supplied split: 33 objects
are in `trainingset.npy` and three are in `testset.npy`. Consequently, results
on these 36 objects must be described as a **common demo evaluation subset**,
not as performance on the official held-out test split. A claim of
"object-level test-set evaluation" requires rerunning every method on the 388
IDs in `testset.npy`.

## 2. Model configuration

### 2.1 Geometry-reasoning VLM

| Item | Configuration |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Fine-tuned output | `qwen-vl-finetune/output_covetwin_7b`, conventionally copied to `pretrain/covetwin_vlm` |
| Parameter update | Language backbone, LM head, and multimodal merger are trainable; vision encoder is frozen; no LoRA/adapters |
| Numeric precision | BF16 |
| Attention | FlashAttention 2 |
| Stored input renders | 1,024 x 1,024 RGBA |
| VLM preprocessing | area constrained to 65,536--262,144 pixels, i.e. 256 x 256--512 x 512 for square inputs |
| Inference image | explicitly resized to 512 x 512 RGB before processor input |
| Maximum sequence length | 8,192 tokens |
| Coarse voxel resolution | `R = 32`, hence at most `32^3 = 32,768` occupied cells |
| Candidate count | `K = 5` per part |

"Full fine-tuning" in this repository means full-parameter supervised
fine-tuning of the trainable language and multimodal-merger modules, rather
than parameter-efficient adaptation. It does **not** mean that the vision tower
is updated: `--tune_mm_vision False` freezes it.

The legacy checkpoint under `pretrain/vlm` comes from the PhysX-Anything
checkpoint package and uses the legacy geometry response format. It must not be
reported as the CoVeTwin relative-shape-span checkpoint unless it has actually
been replaced by the fine-tuned weights.

### 2.2 Coarse-to-fine flow decoder

The production decoder combines the controlled sparse-structure checkpoint
from `Caoza/PhysX-Anything` with modules from
`microsoft/TRELLIS-image-large`. The deployed pipeline is defined by
`pretrain/decoder/pipeline.json`.

| Item | Configuration |
|---|---|
| Coarse control grid | 32 x 32 x 32, centered in a 64 x 64 x 64 decoder grid |
| Sparse latent resolution | 16 |
| Image conditioning encoder | DINOv2 ViT-L/14 register model |
| Sparse-structure sampler | controlled Flow Euler sampler, 25 steps |
| Structured-latent sampler | Flow Euler sampler, 25 steps |
| Total flow updates per asset | 50, reported as 25 + 25 rather than one 50-step sampler |
| CFG strength | 5.0 for both samplers |
| CFG interval | `[0.5, 1.0]` |
| Time rescaling | 3.0 |
| Minimum sigma | `1e-5` |
| Export simplification | PyVista target reduction `0.5` (remove approximately 50% of faces and retain approximately 50%) |
| Texture size | 1,024 x 1,024 |

The controlled-flow training configuration in
`configs/generation/ss_flow_img_dit_L_16l8_fp16.json` initializes from the
TRELLIS sparse-structure denoiser, freezes the base denoiser, and updates the
control branch. It uses AdamW with learning rate `1e-4`, zero weight decay,
four samples per GPU, FP16, EMA rate 0.9999, classifier-free dropout 0.1, and an
adaptive gradient-norm cap of 1.0. The configuration allows 1,000,000 updates;
the deployed filename `denoiser_step0350000.pt` identifies the 350,000-step
checkpoint. This is iteration-based training, so an epoch count is not defined
for the flow-control stage.

## 3. VLM fine-tuning protocol

The author-reported training used **four NVIDIA A800 GPUs** and full
fine-tuning of the trainable VLM modules described above. The exact command is:

```bash
cd qwen-vl-finetune

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export COVETWIN_ANNOTATION_PATH=../dataset/covetwin_training/conversations_train.json
export COVETWIN_IMAGE_ROOT=../dataset_toolkits/renders_all
export OUTPUT_DIR=./output_covetwin_7b

bash scripts/run_sft_covetwin.sh
```

The effective training configuration is:

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW (`adamw_torch`) |
| Adam coefficients | beta1 = 0.9, beta2 = 0.999, epsilon = `1e-8` |
| Learning rate | `2e-5` |
| Weight decay | `0` |
| LR schedule | cosine |
| Warm-up ratio | `0.03` |
| Epochs | 30 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| GPUs | 4 |
| Effective global batch size | 32 |
| Maximum gradient norm | 1.0 |
| DeepSpeed | ZeRO stage 3 |
| Gradient checkpointing | enabled |
| Dataloader workers | 8 per process |
| Save interval | 300 optimizer steps |
| Retained checkpoints | 1 |

The effective batch size is

\[
B_{\mathrm{global}} = B_{\mathrm{device}}\,N_{\mathrm{GPU}}\,
N_{\mathrm{accum}} = 1\times4\times8=32.
\]

## 4. Inference protocol and randomness

The paper-facing default inference configuration is:

| Component | Seed / sampling rule |
|---|---|
| Global VLM response | deterministic decoding (`do_sample=False`) |
| Part geometry candidates | `K=5`, temperature 0.7, top-p 0.9 |
| Base inference seed | 2026 |
| Per-object seed | stable SHA-256-derived offset from the sample ID |
| Per-part/candidate seed | deterministic offset from object seed, part index, and candidate index |
| Flow decoder | seed 2026 through `run_covetwin.py` |
| Metric surface sampling/alignment | seed 2026 with stable per-object offsets |
| VLM training seed | 42 (`SEED`), recorded in `seeds.json`; dataloader shuffle seeded by `COVETWIN_DATA_SEED` (default 42) |

All reported tables currently correspond to **one training run and one
inference run per method** (`n=1`). The five candidates are samples within one
inference run and are not five independent experimental repetitions. Dataset
standard deviations produced by the evaluator measure variation across
objects, not variation across training seeds. If mean +/- standard deviation
over runs is claimed, at least three independently trained and evaluated seeds
must be run and reported.

The training dataloader shuffle is explicitly seeded: `qwenvl/data/data_qwen.py`
and `data_qwen_packed.py` draw from `random.Random(resolve_data_seed(...))`,
where the seed resolves with priority `--data_seed` argument, then the
`COVETWIN_DATA_SEED` environment variable, then the default 42.
`qwenvl/train/train_qwen.py` additionally calls `transformers.set_seed` and
writes `seeds.json` (both `seed` and `data_seed`) into the output directory,
and `scripts/run_sft_covetwin.sh` forwards `SEED` and `COVETWIN_DATA_SEED`
(both defaulting to 42). Training-order randomness is therefore reproducible
for a fixed seed, hardware configuration and dataloader-worker count.

## 5. Relative shape-span representation

Let the occupied voxel set for part $i$ be

\[
\mathcal V_i\subseteq\{0,\ldots,R-1\}^3,
\]

and linearize each voxel with

\[
\pi(x,y,z)=xR^2+yR+z.
\]

After sorting the occupied scalar indices, maximal consecutive runs form
absolute spans $[s_m,e_m]$, $m=1,\ldots,M_i$. CoVeTwin sets

\[
b_i=s_1,\qquad \Delta_m=s_m-b_i,\qquad \ell_m=e_m-s_m+1,
\]

and serializes ordinary text in the form

```text
rss b_i Delta_1:length_1 Delta_2:length_2 ...
```

No additional vocabulary entries or independent 3D tokenizer are introduced.

### 5.1 Numerical serialization cost

For a non-negative integer $u$, define its decimal-symbol cost as

\[
d(u)=
\begin{cases}
1, & u=0,\\
1+\lfloor\log_{10}u\rfloor, & u>0.
\end{cases}
\]

The absolute-span numerical cost is

\[
C_i^{\mathrm{abs}}=
\sum_{m=1}^{M_i}\left[d(s_m)+d(e_m)\right],
\]

whereas relative shape-span compression has cost

\[
C_i^{\mathrm{rel}}=d(b_i)+
\sum_{m=1}^{M_i}\left[d(\Delta_m)+d(\ell_m)\right].
\]

The corresponding compression ratio is

\[
\rho_i=1-\frac{C_i^{\mathrm{rel}}}{C_i^{\mathrm{abs}}}.
\]

This analytical cost counts only decimal digits and intentionally excludes the
fixed `rss` prefix and delimiters. It characterizes numerical serialization
compactness; it is not identical to the VLM tokenizer length. The `Token Len.`
column in an ablation table should be measured with the same released Qwen
tokenizer for every representation, while $\rho_i$ may be reported separately
as the digit-level compression ratio.

Because $\Delta_m$ and $\ell_m$ describe local geometric variation instead of
repeatedly specifying positions across the full voxel-index range, they are
usually shorter than absolute boundaries.

### 5.2 Exact reversibility

The absolute boundaries are recovered by

\[
\hat{s}_m=b_i+\Delta_m,\qquad
\hat{e}_m=b_i+\Delta_m+\ell_m-1.
\]

The complete occupied index set is

\[
\hat Q_i=\bigcup_{m=1}^{M_i}
\left\{b_i+\Delta_m+t\mid t=0,\ldots,\ell_m-1\right\}.
\]

Each scalar index is mapped back to its voxel coordinate by

\[
x=\left\lfloor\frac{q}{R^2}\right\rfloor,\qquad
y=\left\lfloor\frac{q\bmod R^2}{R}\right\rfloor,\qquad
z=q\bmod R.
\]

Therefore,

\[
\hat{\mathcal V}_i=\pi^{-1}(\hat Q_i)=\mathcal V_i.
\]

The implementation additionally rejects an empty sequence, out-of-range
indices, non-positive lengths, overlapping spans, non-maximal adjacent spans,
and a nonzero first offset. Exact round-trip behavior is covered by the codec
unit tests.

## 6. Candidate verification

Each candidate is first checked against six named validity rules
(`covetwin/verification.py`): `parse_failure`, `empty_voxel_set`,
`non_positive_length`, `non_monotonic_span_order`,
`overlapping_or_duplicate_spans`, and `out_of_grid_indices`. A candidate that
fails any rule is invalid and receives no ranking.

For candidate $k$, let $n_k$ denote occupied voxels, $c_k$ the number of
6-connected components, and $\rho_k$ the fraction of occupied voxels in its
largest component. The default selection rule `connectivity` ranks the valid
candidates lexicographically on

\[
(\rho_k,\;-c_k,\;n_k),
\]

with no scalar weights; remaining ties keep the earliest generation index.
The selection rule is configurable through
`select_candidate(..., selection_rule=...)` and the CLI flags
`--selection-rule` / `--selection-seed` (`run_covetwin.py` and
`covetwin/inference.py`):

| Rule | Behavior |
|---|---|
| `connectivity` | Default; the lexicographic ranking above |
| `connectivity_weighted` | Legacy scalar score $Q_k=100\rho_k-2c_k+\min(n_k,R^3)/R^3$, kept for backward compatibility |
| `first` | First sampled candidate, valid or not; the explicit definition of the "without verification" ablation (`--no-verify-candidates` maps to it) |
| `first_valid` | First candidate passing all validity rules |
| `random_valid` | Uniform draw among valid candidates, seeded by `--selection-seed` (defaults to `--seed`) |
| `likelihood` | Highest VLM log-probability among valid candidates; falls back to `first_valid` when no scores are provided |

If every candidate is invalid, the parseable candidate with the most occupied
voxels is selected and marked `fallback=True` with
`fallback_reason=FALLBACK_MOST_OCCUPIED`; if no candidate parses, an
empty-geometry sentinel (`selected_index=-1`, `FALLBACK_EMPTY_SENTINEL`) is
returned and the part is skipped during merging. An empty candidate list
yields `FALLBACK_NO_CANDIDATES`. `candidate_verification.json` records the
selection rule and seed at the top level and the per-part validity,
selection, fallback, and skipped fields.

The `w/o Verification` ablation keeps the same $K$, VLM, and flow decoder but
uses `--selection-rule first` (equivalently `--no-verify-candidates`), i.e.
candidate zero without validity filtering or ranking.

## 7. Geometry evaluation

### 7.1 Surface preparation and alignment

The evaluator samples 50,000 points uniformly by triangle area from each
predicted and GT surface. Prediction and GT are independently centered by their
axis-aligned bounding-box centers and divided by their respective largest
bounding-box extent. Geometry errors are therefore scale invariant; metric
scale is evaluated separately.

For methods with different coordinate conventions, the prediction is rigidly
aligned to GT. All 24 proper axis-aligned cube rotations are scored, the best
six initialize 20 iterations of 90%-trimmed ICP, and the alignment with the
lowest symmetric L1 nearest-neighbor distance is retained. No reflection or
free scale is allowed.

### 7.2 Chamfer Distance

Let $P$ and $G$ be the aligned normalized prediction and GT point sets, and
let

\[
d(p,G)=\min_{g\in G}\lVert p-g\rVert_2.
\]

The reported squared L2 Chamfer Distance is

\[
\operatorname{CD}(P,G)=
\frac{1}{|P|}\sum_{p\in P}d(p,G)^2+
\frac{1}{|G|}\sum_{g\in G}d(g,P)^2.
\]

There is no factor of $1/2$. The paper displays
$10^3\operatorname{CD}$. CD is computed per object and then macro-averaged
over evaluable objects. The evaluator also stores an L1 diagnostic, but it is
not the paper's primary CD.

### 7.3 F-score

The fixed threshold is

\[
\tau=0.05
\]

in the largest-extent-normalized coordinate system. Precision and recall are

\[
P_\tau=\frac{1}{|P|}\sum_{p\in P}\mathbf 1[d(p,G)\leq\tau],\qquad
R_\tau=\frac{1}{|G|}\sum_{g\in G}\mathbf 1[d(g,P)\leq\tau],
\]

and

\[
F_\tau=\frac{2P_\tau R_\tau}{P_\tau+R_\tau},
\]

with $F_\tau=0$ when $P_\tau+R_\tau=0$. Scores are computed per object and
macro-averaged over evaluable objects.

### 7.4 PSNR

Predicted zero-pose meshes are rendered from the 25 GT camera poses using
Blender Cycles with 128 samples per pixel. The output resolution matches the GT
render (1,024 x 1,024 in the supplied data), and both RGBA images are composited
over white. For a view with RGB values in $[0,1]$,

\[
\operatorname{MSE}=\frac{1}{3HW}\sum_{h,w,c}
(I_{h,w,c}-\hat I_{h,w,c})^2,
\]

\[
\operatorname{PSNR}=10\log_{10}\frac{1}{\operatorname{MSE}}.
\]

An exact or numerically zero MSE is capped at 100 dB. The primary paper value
is full-frame PSNR: the arithmetic mean over views is computed for each object,
then object values are macro-averaged. A foreground-union PSNR using
`alpha > 0.01` is saved only as a diagnostic. Missing renders are not assigned
zero; they are reported unavailable and excluded, so the metric count must
always accompany the mean.

### 7.5 Absolute scale error

Let $\hat{\mathbf d},\mathbf d\in\mathbb R^3$ be predicted and annotated
dimensions in centimeters, sorted in descending order to remove axis-order
ambiguity. The primary scale error is

\[
E_{\mathrm{scale}}=\lVert
\operatorname{sort}(\hat{\mathbf d})-
\operatorname{sort}(\mathbf d)\rVert_2\quad [\mathrm{cm}].
\]

It is computed per object and macro-averaged. For a method that does not report
metric dimensions, the evaluator applies a GT-largest-dimension calibration to
raw mesh extents and labels the value as a mesh-coordinate proxy. Reported
dimensions and proxy-derived dimensions must not be presented as having the
same provenance without this qualification.

## 8. Articulation evaluation

Predicted parts are matched to GT parts using aligned per-part surface Chamfer
cost and Hungarian assignment; label/name matching is only a fallback. Joint
groups are then matched by Hungarian assignment on the IoU of their matched
child-part sets. Axis, origin, and range errors are evaluated only for matched
groups with the correct joint type and compatible degree-of-freedom slot.

### 8.1 Joint-type accuracy

For object $o$, unmatched predicted and GT groups are errors, and

\[
A_o=\frac{N_o^{\mathrm{correct}}}
{\max(N_o^{\mathrm{pred}},N_o^{\mathrm{gt}})}.
\]

The main value is the object-macro mean of $A_o$. The evaluator also stores a
pooled correct/denominator accuracy; the two aggregation schemes must be named
explicitly rather than interchanged.

### 8.2 Axis error

For unit axes $\hat{\mathbf a}$ and $\mathbf a$, the direction-sign-invariant
angular error is

\[
E_{\mathrm{axis}}=
\arccos\!\left(\left|\hat{\mathbf a}^{\mathsf T}\mathbf a\right|\right)
\frac{180}{\pi}\quad [^\circ].
\]

There is no correctness threshold or clipping beyond numerical dot-product
clamping. Errors are averaged over valid joints within each object and then
macro-averaged over objects.

### 8.3 Origin error

For a revolute joint, origin error is the shortest distance between the
predicted and GT axis lines. With origins $\hat{\mathbf o},\mathbf o$,

\[
E_{\mathrm{origin}}=
\begin{cases}
\dfrac{|(\hat{\mathbf o}-\mathbf o)^{\mathsf T}
(\hat{\mathbf a}\times\mathbf a)|}
{\lVert\hat{\mathbf a}\times\mathbf a\rVert_2},
& \lVert\hat{\mathbf a}\times\mathbf a\rVert_2>\epsilon,\\[6pt]
\lVert(\hat{\mathbf o}-\mathbf o)\times\mathbf a\rVert_2,
& \text{otherwise}.
\end{cases}
\]

For a prismatic joint it is

\[
E_{\mathrm{origin}}=\lVert\hat{\mathbf o}-\mathbf o\rVert_2.
\]

Normalized distances are restored to metric scale with the annotated GT
maximum dimension and reported in centimeters. No success cutoff is applied.
The per-object and then object-macro averaging rule is the same as for axis
error.

### 8.4 Unitless motion-range error

Continuous joints are excluded because they do not have finite motion limits.
For every matched, type-correct, non-continuous revolute or prismatic joint,
define the positive physical range

\[
r_j=u_j-l_j,\qquad \hat r_j=\hat u_j-\hat l_j,
\]

after converting both prediction and GT to a common native unit: radians for a
revolute joint and meters for a prismatic joint. The unitless error normalizes
the absolute span difference by the joint family's natural scale: $\pi$ for a
revolute joint and $D_o$, the annotated largest object dimension, for a
prismatic joint:

\[
E_{\mathrm{range},j}=
\begin{cases}
|\hat r_j-r_j|/\pi, & \text{revolute},\\[2pt]
|\hat r_j-r_j|/D_o, & \text{prismatic}.
\end{cases}
\]

Axis disagreement is deliberately not included in this quantity because it is
already measured by $E_{\mathrm{axis}}$. The primary score first averages
valid joints within each object and then macro-averages the object scores.
There is no threshold or upper clipping. The metric count must be reported
because objects without a matched finite-range joint are excluded. Per-record
native-unit diagnostics are stored as `revolute_range_error_rad` and
`prismatic_range_error_m`. This is the default of
`evaluation/evaluate_metrics.py`; `--legacy-range-error` reproduces the
superseded axis-weighted vector-norm variant.

## 9. Physical attributes and execution

Material strings are lowercase-canonicalized labels. Affordance is evaluated as
the discrete `priority_rank` label. For each label $c$,

\[
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c},
\qquad
F1_{\mathrm{macro}}=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}F1_c.
\]

Missing and spurious parts contribute false negatives and false positives.
The paper uses object-level macro-F1 followed by an object-macro mean. Methods
that do not export a discrete label are `N/A` unless the explicit
`--score-missing-semantics-as-zero` policy is enabled.

For MuJoCo execution rate, a missing output is a failure. A generated asset is
successful when it can be standardized to MJCF, compiled by MuJoCo, initialized
at finite in-range hinge/slide coordinates, forwarded, and stepped 100 times
without an exception or non-finite `qpos`, `qvel`, or `qacc`. The standardized
URDF adapter may add or repair required inertial defaults but does not alter the
generated geometry, link graph, joint type, axis, origin, or limits. For $N$
objects,

\[
\operatorname{ExecutionRate}=\frac{1}{N}
\sum_{o=1}^{N}\mathbf 1[\text{asset }o\text{ succeeds}].
\]

The paper-facing engine is MuJoCo. The unified geometry evaluator additionally
provides an isolated PyBullet diagnostic; those numbers must not be labeled as
MuJoCo results.

## 10. Hardware and software environment

| Component | Version / configuration |
|---|---|
| Training accelerators | **4 x NVIDIA A800** (**reported**) |
| GPU memory | Not retained in the repository or accessible through NVML during this audit; the exact 40 GB or 80 GB A800 SKU must be confirmed before submission |
| CPU on inspected project host | 2 x Intel Xeon Gold 6133 at 2.50 GHz, 40 physical cores / 80 threads |
| Host RAM | 251 GiB usable |
| Operating system | Ubuntu 20.04.6 LTS, Linux kernel 5.15.0-139-generic, x86-64 |
| Python | 3.10.20 |
| CUDA toolkit | 11.8 (`nvcc` 11.8.89) |
| PyTorch | 2.1.1+cu118 |
| Transformers | 4.50.0 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Trimesh | 4.0.5 |
| MuJoCo | 3.10.0 |
| PyBullet diagnostic | 3.2.7 |
| Blender renderer | Bundled Blender 3.6 series; use this rather than system Blender 2.82 for modern CUDA architectures |

The GPU driver version, exact A800 memory capacity, and a machine-readable
training checkpoint hash were not found in the retained artifacts. These three
fields should be captured from the original training machine rather than
guessed. A suitable hardware capture command is:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sha256sum qwen-vl-finetune/output_covetwin_7b/model*.safetensors
```

## 11. Reproduction commands

Run the full four-stage inference pipeline with the paper defaults:

```bash
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python run_covetwin.py \
  --demo-path demo \
  --output-path test_covetwin \
  --ckpt pretrain/covetwin_vlm \
  --candidate-count 5 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 2026 \
  --verify-candidates \
  --stages 1 2 3 4
```

Run the unified metrics with the fixed geometry and render protocol:

```bash
python evaluate_covetwin_metrics.py \
  --pred-roots test_covetwin \
  --dataset-root dataset/PhysX_mobility \
  --renders-root dataset_toolkits/renders_all \
  --surface-samples 50000 \
  --alignment-samples 4000 \
  --icp-iterations 20 \
  --icp-candidates 6 \
  --fscore-threshold 0.05 \
  --seed 2026 \
  --render-engine CYCLES \
  --render-device GPU \
  --render-samples 128 \
  --output-dir evaluation_results/covetwin
```

Run the MuJoCo execution benchmark separately:

```bash
python tools/build_mujoco_execution_suite.py \
  --output mujoco \
  --steps 100 \
  --overwrite
```

## 12. Statistics and auditability

Paper-facing comparison and aggregation tables must be regenerated from the
released per-object outputs with the auditable tooling in `evaluation/`,
not transcribed by hand:

- **`evaluation/paired_statistics.py`** compares two methods on the objects
  both completed, pairing by `gt_id`. It reports per-metric paired bootstrap
  95% confidence intervals (`--bootstrap`, default B = 10,000) and two-sided
  sign-flip permutation p-values under the Monte Carlo convention
  `p = (e + 1) / (B + 1)` (`--mct`, default B = 10,000), with Holm step-down
  correction across the metric family. The JSON report embeds full-precision
  values, SHA256 digests of every input file, the seed and the UTC timestamp;
  identical inputs and seed reproduce identical statistics. CSV and an
  optional IEEE LaTeX fragment (`--emit-latex`) are emitted alongside.
- **`evaluation/aggregate_runs.py`** aggregates repeated runs (per-run
  per-object CSVs or directories) into per-method per-metric run means and
  then across-run mean +/- sample standard deviation (`ddof=1`), so run-level
  variance is reported honestly instead of a single pooled number. Outputs are
  JSON (with per-input SHA256 digests), an aggregate CSV, a run-level CSV and
  an optional LaTeX fragment (`--emit-latex`).
- **`evaluation/efficiency_profile.py`** profiles token counts of geometry
  strings (per-part and aggregate mean/median/P95/max) with the released
  HuggingFace tokenizer or a deterministic regex fallback counter, and
  provides wall-time and peak-CUDA-memory profiling helpers.

Any mean +/- standard deviation over runs in the paper must come from at
least three independently seeded runs aggregated this way, and every
statistics table must cite the seed and input digests recorded in the
corresponding JSON report.

## 13. Implementation audit and camera-ready actions

The following earlier discrepancies between a strict paper protocol and the
repository have been re-audited against the current code:

1. **Split enforcement — RESOLVED.** `training/build_dataset.py` now defaults
   to `--split train`, which restricts output to the 1,636 IDs in
   `dataset/splits/trainingset.npy` and excludes the 388 test objects.
   `--split all` hard-errors without the explicit `--allow-test-leak` opt-in,
   `--split-dir`/`--split-json` allow split overrides, and
   `--make-split-json`/`--split-seed`/`--test-count` deterministically
   regenerate a split. See Section 1.2 for the train-only command.
2. **Current 36-object table — KNOWN LIMITATION (not resolved).** The demo
   benchmark remains 33 training objects plus three test objects, not a
   held-out test set. This is a reporting caveat of the existing demo tables,
   not a code defect: those tables must be labeled as a common demo evaluation
   subset, and any claim of object-level test-set performance requires
   rerunning every method on the 388 IDs in `testset.npy`.
3. **Motion-range implementation — RESOLVED.**
   `evaluation/evaluate_metrics.py` now implements the Section 8.4 definition
   by default: continuous joints are excluded, revolute joints report
   `|span error| / pi` and prismatic joints `|span error| / D_o`, with
   per-record `revolute_range_error_rad` / `prismatic_range_error_m`
   diagnostics in native units. The superseded axis-weighted vector-norm
   variant remains available only behind `--legacy-range-error`.
4. **Training randomness — RESOLVED.** The dataloader shuffle in
   `qwenvl/data/data_qwen.py` and `data_qwen_packed.py` uses
   `random.Random(resolve_data_seed(...))` with priority `--data_seed` >
   `COVETWIN_DATA_SEED` > 42; `qwenvl/train/train_qwen.py` calls
   `transformers.set_seed` and writes `seeds.json`;
   `scripts/run_sft_covetwin.sh` forwards `SEED` and `COVETWIN_DATA_SEED`
   (both default 42).
5. **Run metadata and multi-seed claims — RESOLVED (tooling); hardware capture
   still open.** Multi-seed mean +/- standard deviation claims are now
   supported by `evaluation/aggregate_runs.py` (across-run sample standard
   deviation, `ddof=1`) and paired comparisons by
   `evaluation/paired_statistics.py`; see Section 12. What remains is to
   capture the A800 memory SKU, GPU driver and a training-checkpoint hash
   from the original training machine (Section 10 gives the commands) and to
   actually run the additional seeds before making a multi-seed claim.
