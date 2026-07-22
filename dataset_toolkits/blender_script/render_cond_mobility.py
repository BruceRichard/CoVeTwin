import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
import ipdb
import glob



def render(savepath,file_path,rotate=0,num_views=25):
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    with open("views.json", "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=4)

    
    # Use the Blender binary from $BLENDER if set, else fall back to system 'blender'.
    # System Blender 2.82 cannot compile CUDA kernels for RTX 4090 (sm_89); point
    # $BLENDER at the bundled blender-3.6 build instead.
    blender_bin = os.environ.get('BLENDER', 'blender')
    os.system('{} --background --python blender_script/render_mobility.py --   --object {} --rotate {} --output_folder {}'.format(blender_bin, os.path.expanduser(file_path),rotate,savepath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=4000)
    parser.add_argument("--datapath", type=str, default='../dataset/PhysX_mobility/partseg',
                        help='Directory containing one subfolder per object (each with objs/).')
    parser.add_argument("--basepath", type=str, default='../dataset/PhysX_mobility/renders',
                        help='Output directory: renders are written to <basepath>/<object_id>/.')
    args = parser.parse_args()


    datapath=args.datapath

    # sorted() makes the shard split deterministic across processes; without it
    # os.listdir order is not guaranteed and shards could overlap or skip objects.
    namelist=sorted(os.listdir(datapath))
    namelist=namelist[args.index*args.range:(args.index+1)*args.range]
    basepath=args.basepath
    os.makedirs(basepath, exist_ok=True)

    for name in namelist:
        
        savepath=os.path.join(basepath,name)
        os.makedirs(savepath, exist_ok=True)
        if not os.path.exists(os.path.join(savepath,'transforms.json')):

                objpath=os.path.join(datapath,name)
            
                render(savepath,objpath)

