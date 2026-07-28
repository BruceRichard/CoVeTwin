# CoVeTwin AAAI Media Supplement

**Structure-Aware Geometry Compression and Verification for High-Fidelity
Articulated Digital Twin Generation from a Single Image**

Open `index.html` in a modern browser to view the complete offline gallery.
No web server, network connection, model checkpoint, or external stylesheet is
required.

## What this supplement demonstrates

1. **Method and representation.** The pipeline overview explains compact
   relative occupancy-span prediction, candidate verification, coarse-to-fine flow
   reconstruction, part transfer, and simulator-ready export.
2. **Ablation evidence.** The qualitative ablation compares direct voxel,
   absolute index, absolute span, no-verification, and full CoVeTwin outputs.
3. **Physical-property binding.** Mesh-aligned affordance and material panels
   demonstrate that physical attributes are assigned to the refined part
   meshes rather than to coarse voxel previews.
4. **Articulation playback.** Seven benchmark objects show generated textured
   geometry and predicted movable joints under PyBullet actuation.
5. **Real-world generalization.** Three selected real-image examples show
   articulated reconstruction beyond the curated benchmark renderings.
6. **Cross-method dynamics.** Seven common benchmark inputs compare
   Articulate Anything, PhysX-3D, PhysX-Anything, URDF-Anything, and CoVeTwin
   under the same playback interface. A genuinely missing generated output is
   shown as missing rather than substituted.
7. **Dynamic ablation.** Seven inputs compare voxel coordinates, voxel
   indices, absolute spans, relative occupancy spans without verification, and
   full CoVeTwin over complete joint motion.

## Directory layout

```text
media_supplement_aaai/
├── index.html
├── README.md
├── 01_method/                  # Method and compression figures
├── 02_ablation/                # Main ablation figure and detailed globe maps
├── 03_physical_attributes/     # Input / affordance / material triplets
├── 04_articulation_benchmark/  # Benchmark input-video pairs
├── 05_real_world/              # Real-image input-video pairs
├── 06_cross_method_dynamics/   # Seven common-input method comparisons
└── 07_dynamic_ablation/        # Seven controlled ablation comparisons
```

## Media conventions

- All videos are MP4 files encoded with H.264, use a 512 x 512 frame, and can
  be played with standard browser controls.
- Videos show PyBullet articulation playback of the generated URDF assets.
  They are qualitative demonstrations, not the MuJoCo execution-rate metric.
- The cross-method and ablation rows preserve a common input within each row.
  Method names are attached to the corresponding retained outputs; no video is
  relabeled or used to fill a missing method result.
- Affordance maps use a normalized 0--1 scale. Material maps visualize
  material-dependent log10 Young's modulus over the range -3--3 GPa.
- Transparent heatmap backgrounds are displayed over a light neutral panel by
  the offline gallery.
- The benchmark inputs originate from the PhysX-Mobility evaluation assets.
  The real-world inputs are the project's selected wild-image demonstrations.

## Evaluation scope

The paper-facing dataset split contains 1,636 training objects and 388 test
objects. With 25 rendered views per object, this gives 40,900 training views
and 9,700 test views. The media gallery is a qualitative subset selected from
those available renderings; it is not used to recompute the paper's aggregate
metrics. The separately retained 36-object local audit is a reproducibility
diagnostic and must not be substituted for the official test-split results.

## Recommended viewing order

Start with the method overview and static ablation, inspect the physical
attribute triplets, and then use the cross-method and dynamic-ablation grids for
the strongest direct evidence. Finish with the standalone benchmark and
real-world examples. The entire package is self-contained and kept below the
AAAI 50 MB media-supplement limit.
