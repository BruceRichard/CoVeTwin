#!/usr/bin/env python3
"""Blender worker used by evaluation/evaluate_metrics.py.

This file intentionally uses only Blender-bundled modules.  It reproduces the
camera, lighting, transparent-film and normalization setup used by
dataset_toolkits/blender_script/render_mobility.py, while accepting prediction
OBJ files in the nested ``objs/<part>/<part>.obj`` layout.
"""

import argparse
import glob
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


WORKER_VERSION = 2


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-root", required=True)
    parser.add_argument("--mesh-manifest")
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES"
    )
    parser.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


def configure_render(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.filter_type = "BOX"
        scene.cycles.filter_width = 1
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 3
        scene.cycles.transmission_bounces = 3
        scene.cycles.use_denoising = True
        scene.cycles.device = args.device
        if args.device == "GPU":
            try:
                preferences = bpy.context.preferences.addons["cycles"].preferences
                preferences.get_devices()
                device_type = os.environ.get("BLENDER_CYCLES_DEVICE", "CUDA")
                preferences.compute_device_type = device_type
                for device in preferences.devices:
                    device.use = True
            except Exception as exc:
                print(
                    "[WARN] unable to configure Cycles GPU, Blender may fall back to CPU:",
                    exc,
                )
    else:
        scene.eevee.taa_render_samples = args.samples


def import_mesh(path):
    extension = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if extension == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError("unsupported render mesh format: {}".format(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def load_prediction(object_root, mesh_manifest=None):
    if mesh_manifest:
        with open(mesh_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        visuals = manifest.get("visuals", [])
        if not visuals:
            raise RuntimeError("mesh manifest contains no visuals")
        paths = []
        for visual in visuals:
            path = visual["path"]
            print("Loading", path)
            imported = import_mesh(path)
            transform = Matrix(visual.get("transform", Matrix.Identity(4)))
            for obj in imported:
                obj.matrix_world = transform @ obj.matrix_world
            paths.append(path)
        return paths

    paths = sorted(
        glob.glob(os.path.join(object_root, "objs", "**", "*.obj"), recursive=True)
    )
    if not paths:
        raise RuntimeError("no OBJ files under {}/objs".format(object_root))
    for path in paths:
        print("Loading", path)
        import_mesh(path)
    return paths


def scene_bbox():
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        found = True
        for coordinate in obj.bound_box:
            world = obj.matrix_world @ Vector(coordinate)
            bbox_min.x = min(bbox_min.x, world.x)
            bbox_min.y = min(bbox_min.y, world.y)
            bbox_min.z = min(bbox_min.z, world.z)
            bbox_max.x = max(bbox_max.x, world.x)
            bbox_max.y = max(bbox_max.y, world.y)
            bbox_max.z = max(bbox_max.z, world.z)
    if not found:
        raise RuntimeError("no mesh objects in Blender scene")
    return bbox_min, bbox_max


def normalize_and_align(alignment):
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    root = bpy.data.objects.new("EvaluationRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in roots:
        obj.parent = root

    bbox_min, bbox_max = scene_bbox()
    extent = bbox_max - bbox_min
    scale = 1.0 / max(extent.x, extent.y, extent.z)
    root.scale = root.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    root.matrix_world.translation += -(bbox_min + bbox_max) / 2.0
    bpy.context.view_layer.update()

    rotation = alignment["rotation"]
    translation = alignment["translation"]
    matrix = Matrix(
        (
            (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
            (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
            (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    root.matrix_world = matrix @ root.matrix_world
    bpy.context.view_layer.update()
    return scale


def create_camera():
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.sensor_height = 32
    camera.data.sensor_width = 32
    return camera


def create_lights():
    key = bpy.data.objects.new(
        "Default_Light", bpy.data.lights.new("Default_Light", type="POINT")
    )
    bpy.context.collection.objects.link(key)
    key.data.energy = 1000
    key.location = (4, 1, 6)

    top = bpy.data.objects.new(
        "Top_Light", bpy.data.lights.new("Top_Light", type="AREA")
    )
    bpy.context.collection.objects.link(top)
    top.data.energy = 10000
    top.location = (0, 0, 10)
    top.scale = (100, 100, 100)

    bottom = bpy.data.objects.new(
        "Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA")
    )
    bpy.context.collection.objects.link(bottom)
    bottom.data.energy = 1000
    bottom.location = (0, 0, -10)


def expected_manifest(args, frame_count, alignment):
    manifest = {
        "worker_version": WORKER_VERSION,
        "object_root": os.path.realpath(args.object_root),
        "camera_json": os.path.realpath(args.camera_json),
        "engine": args.engine,
        "device": args.device,
        "samples": args.samples,
        "resolution": args.resolution,
        "frame_count": frame_count,
        "alignment": alignment,
    }
    if args.mesh_manifest:
        with open(args.mesh_manifest, "r", encoding="utf-8") as handle:
            mesh_manifest = json.load(handle)
        manifest["render_visuals"] = mesh_manifest.get("visuals", [])
    return manifest


def main():
    args = arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.camera_json, "r", encoding="utf-8") as handle:
        cameras = json.load(handle)
    with open(args.alignment_json, "r", encoding="utf-8") as handle:
        alignment = json.load(handle)
    frames = cameras.get("frames", [])
    if not frames:
        raise RuntimeError("camera JSON contains no frames")

    if args.force:
        for path in glob.glob(os.path.join(args.output_dir, "[0-9][0-9][0-9].png")):
            os.unlink(path)
        manifest_path = os.path.join(args.output_dir, "render_manifest.json")
        if os.path.exists(manifest_path):
            os.unlink(manifest_path)

    clear_scene()
    configure_render(args)
    load_prediction(args.object_root, args.mesh_manifest)
    normalize_and_align(alignment)
    camera = create_camera()
    create_lights()

    for index, frame in enumerate(frames):
        output_path = os.path.join(args.output_dir, "{:03d}.png".format(index))
        if os.path.exists(output_path) and not args.force:
            continue
        camera.matrix_world = Matrix(frame["transform_matrix"])
        camera.data.lens = 16.0 / math.tan(float(frame["camera_angle_x"]) / 2.0)
        bpy.context.scene.render.filepath = output_path
        bpy.context.view_layer.update()
        print("Rendering", output_path)
        bpy.ops.render.render(write_still=True)

    with open(
        os.path.join(args.output_dir, "render_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(expected_manifest(args, len(frames), alignment), handle, indent=2)


if __name__ == "__main__":
    main()
