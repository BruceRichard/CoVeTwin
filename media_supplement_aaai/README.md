# CoVeTwin AAAI Media Supplement

Open `index.html` in a modern browser to view the complete offline gallery.
No web server, network connection, model checkpoint, or external stylesheet is
required.

## What this supplement demonstrates

1. **Method and representation.** The pipeline overview explains compact
   relative shape-span prediction, candidate verification, coarse-to-fine flow
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

## Directory layout

```text
media_supplement_aaai/
├── index.html
├── README.md
├── 01_method/                  # Method and compression figures
├── 02_ablation/                # Main ablation figure and detailed globe maps
├── 03_physical_attributes/     # Input / affordance / material triplets
├── 04_articulation_benchmark/  # Benchmark input-video pairs
└── 05_real_world/              # Real-image input-video pairs
```

## Media conventions

- All videos are MP4 files encoded with H.264, use a 512 x 512 frame, and can
  be played with standard browser controls.
- Videos show PyBullet articulation playback of the generated URDF assets.
  They are qualitative demonstrations, not the MuJoCo execution-rate metric.
- Affordance maps use a normalized 0--1 scale. Material maps visualize
  material-dependent log10 Young's modulus over the range -3--3 GPa.
- Transparent heatmap backgrounds are displayed over a light neutral panel by
  the offline gallery.
- The benchmark inputs originate from the PhysX-Mobility evaluation assets.
  The real-world inputs are the project's selected wild-image demonstrations.

## Recommended viewing order

Start with the method overview, then inspect the main ablation figure. Continue
to the physical-attribute triplets and finish with the benchmark and real-world
articulation videos. The entire package is intentionally self-contained and
kept below the AAAI 50 MB media-supplement limit.
