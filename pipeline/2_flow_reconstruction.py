#!/usr/bin/env python3
"""Stage 2: reconstruct textured geometry with the conditional flow decoder."""

import os
import argparse
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
# Use expandable memory segments to reduce fragmentation. The slat_decoder_mesh
# on GPU 1 takes ~22 GB; without expandable segments the caching allocator
# fragments and causes OOM on large flexicubes mesh extractions.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "covetwin_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "covetwin_matplotlib")
)
import gc
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from trellis.modules import sparse as sp
import numpy as np
import torch
from types import MethodType


def _module_device(module):
    return next(module.parameters()).device


def _clean_sparse_to(x, device):
    out = sp.SparseTensor(
        feats=x.feats.to(device),
        coords=x.coords.to(device),
        shape=x.shape,
        layout=x.layout,
    )
    out._scale = x._scale
    return out


def _move_mesh_extractor(model, device):
    extractor = getattr(model, "mesh_extractor", None)
    if extractor is None:
        return
    extractor.device = str(device)
    extractor.reg_v = extractor.reg_v.to(device)
    extractor.reg_c = extractor.reg_c.to(device)
    mesh_extractor = getattr(extractor, "mesh_extractor", None)
    if mesh_extractor is not None:
        mesh_extractor.device = str(device)
        for attr, value in vars(mesh_extractor).items():
            if torch.is_tensor(value):
                setattr(mesh_extractor, attr, value.to(device))


def _enable_model_parallel(pipeline):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRELLIS decoding.")

    num_devices = torch.cuda.device_count()
    if num_devices < 4:
        pipeline.cuda()
        print(f"[TRELLIS] Model parallel disabled; using cuda:0 with {num_devices} visible GPU(s).")
        return pipeline

    devices = [torch.device(f"cuda:{idx}") for idx in range(num_devices)]
    placement = {
        "sparse_structure_decoder": devices[0],
        "sparse_structure_encoder": devices[0],
        "sparse_structure_flow_model": devices[0],
        "slat_flow_model": devices[0],
        "image_cond_model": devices[0],
        "slat_decoder_mesh": devices[1],
        "slat_decoder_gs": devices[2],
        "slat_decoder_rf": devices[3],
    }

    for name, model in pipeline.models.items():
        device = placement.get(name, devices[0])
        model.to(device)
        _move_mesh_extractor(model, device)

    # --- Split mesh decoder across two GPUs ---
    # The SLatMeshDecoder forward pass (sparse convs + upsampling) and
    # flexicubes mesh extraction together can exceed 23 GiB. Place the
    # model weights + forward pass on one GPU and flexicubes on another.
    if num_devices >= 5:
        mesh_decoder = pipeline.models.get("slat_decoder_mesh")
        if mesh_decoder is not None and hasattr(mesh_decoder, "mesh_extractor"):
            flex_device = devices[4]
            _move_mesh_extractor(mesh_decoder, flex_device)
            # Patch to_representation so the sparse feature tensor is moved
            # to the flexicubes GPU before dense conversion + mesh extraction.
            # The torch.cuda.device context is essential: factory functions
            # inside flexicubes (torch.zeros, etc.) must land on flex_device.
            orig_to_rep = mesh_decoder.to_representation
            def split_to_representation(self, x):
                x_flex = _clean_sparse_to(x, flex_device)
                with torch.cuda.device(flex_device):
                    return orig_to_rep(x_flex)
            mesh_decoder.to_representation = MethodType(split_to_representation, mesh_decoder)
            print(f"[TRELLIS] Mesh decoder split: forward on {placement['slat_decoder_mesh']}, "
                  f"flexicubes on {flex_device}", flush=True)

    def decode_slat_model_parallel(self, slat, formats=['mesh', 'gaussian', 'radiance_field']):
        ret = {}
        for fmt, model_name in (
            ("mesh", "slat_decoder_mesh"),
            ("gaussian", "slat_decoder_gs"),
            ("radiance_field", "slat_decoder_rf"),
        ):
            if fmt not in formats:
                continue
            model = self.models[model_name]
            device = _module_device(model)
            with torch.cuda.device(device):
                # --- memory debug: GPU state before forward ---
                mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                mem_resvd = torch.cuda.memory_reserved(device) / 1024**3
                mem_free  = mem_total - mem_resvd
                print(f"[MEM] {fmt:>15s} on {device}: "
                      f"alloc={mem_alloc:.2f}G reserved={mem_resvd:.2f}G free={mem_free:.2f}G "
                      f"(total={mem_total:.2f}G)", flush=True)
                slat_on_device = _clean_sparse_to(slat, device)
                ret[fmt] = model(slat_on_device)
            del slat_on_device
            torch.cuda.empty_cache()
        return ret

    pipeline.decode_slat = MethodType(decode_slat_model_parallel, pipeline)
    print("[TRELLIS] Model parallel enabled:")
    for name in sorted(placement):
        print(f"  - {name}: {placement[name]}")
    return pipeline


def decode_prediction_root(
    pipeline, demo_path, prediction_path, seed, simplify, texture_size, only=None
):
    namelist = sorted(os.listdir(demo_path))
    selected = set(only or [])
    print(f"[TRELLIS] Decode root {prediction_path} from images in {demo_path}")
    for name in namelist:
        image_path = os.path.join(demo_path, name)
        if not os.path.isfile(image_path):
            continue
        sample_id = os.path.splitext(name)[0]
        if selected and sample_id not in selected:
            continue
        image = Image.open(image_path)
        qwenpath = os.path.join(prediction_path, sample_id)
        out_glb = os.path.join(qwenpath, 'sample.glb')
        if os.path.exists(out_glb):
            print(f"skip existing: {out_glb}")
            continue

        occupancy_path = os.path.join(qwenpath, 'allind.npy')
        if not os.path.exists(occupancy_path):
            print(f"skip missing occupancy: {occupancy_path}")
            continue
        newcoords = np.load(occupancy_path)

        size = 32
        resolution = 64
        newcoords = newcoords + 32 - (size) // 2

        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, newcoords[:, 0], newcoords[:, 1], newcoords[:, 2]] = 1
        ss = ss.cuda().float().unsqueeze(0)

        with torch.inference_mode():
            outputs = pipeline.run_control(ss, image, seed=seed)

        # Move Gaussian internal tensors to cuda:0. The Gaussian is decoded on
        # cuda:2 (slat_decoder_gs) in model-parallel mode, but the downstream
        # render_multiview creates camera parameters on cuda:0. Feeding cross-device
        # tensors to diff_gaussian_rasterization's CUDA kernel causes
        # "CUDA error: an illegal memory access".
        gaussian = outputs['gaussian'][0]
        for attr in [
            '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
            '_opacity', 'aabb', 'scale_bias', 'rots_bias', 'opacity_bias'
        ]:
            val = getattr(gaussian, attr, None)
            if val is not None and val.device.type == 'cuda':
                setattr(gaussian, attr, val.to('cuda:0'))

        glb = postprocessing_utils.to_glb(
            gaussian,
            outputs['mesh'][0],
            simplify=simplify,
            texture_size=texture_size,
        )

        # to_glb already copied mesh vertices/faces to CPU (numpy). The GPU
        # mesh + gaussian + radiance_field on cuda:1/2/3 are no longer needed.
        del outputs, gaussian
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

        glb.export(out_glb)

        del glb, ss
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Decode one or more CoVeTwin coarse-voxel roots."
    )
    parser.add_argument('--demo_path', type=str, default='./demo_new')
    parser.add_argument('--input_paths', nargs='+', default=['./test_demo_new'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--simplify', type=float, default=0.5)
    parser.add_argument('--texture_size', type=int, default=1024)
    parser.add_argument('--only', nargs='*', default=[])
    args = parser.parse_args()

    # Load once and reuse for every ablation root sharing the same image set.
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        str(PROJECT_ROOT / "pretrain" / "decoder")
    )
    pipeline = _enable_model_parallel(pipeline)
    for input_path in args.input_paths:
        decode_prediction_root(
            pipeline,
            args.demo_path,
            input_path,
            args.seed,
            args.simplify,
            args.texture_size,
            args.only,
        )
