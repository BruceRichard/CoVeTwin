# CoVeTwin pipeline stages

The numbered scripts preserve the four-stage execution order described in the
paper:

1. `1_geometry_reasoning.py` predicts compact part geometry and verifies
   multiple candidates.
2. `2_flow_reconstruction.py` reconstructs high-resolution textured geometry
   with the image-conditioned flow decoder.
3. `3_part_segmentation.py` transfers coarse part labels to the refined mesh.
4. `4_simulation_export.py` exports physical metadata, URDF, and MuJoCo XML.

Use `python run_covetwin.py ...` from the repository root for normal inference.
The stage scripts remain directly executable for debugging and partial runs.
