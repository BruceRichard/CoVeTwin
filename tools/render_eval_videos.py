import os, math, time, sys
import numpy as np
import imageio
import pybullet as p
import pybullet_data
import xml.etree.ElementTree as ET


def urdf_link_name(element):
    """Read URDF parent/child links, including legacy name attributes."""
    if element is None:
        return None
    return element.get('link') or element.get('name')


def validate_urdf_tree(urdf_path):
    """Validate a single-root URDF tree before handing it to PyBullet."""
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return False, 'XML parse failed: ' + str(exc)

    link_list = [link.get('name') for link in root.findall('.//link')]
    if any(name is None for name in link_list):
        return False, 'link without name'
    links = set(link_list)
    if len(links) != len(link_list):
        return False, 'duplicate link name'
    if not links:
        return False, 'no links'

    parent_by_child = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name', '<unnamed>')
        parent = urdf_link_name(joint.find('parent'))
        child = urdf_link_name(joint.find('child'))
        if parent not in links or child not in links:
            return False, f'joint {joint_name} references an unknown link'
        if parent == child:
            return False, f'joint {joint_name} has the same parent and child'
        if child in parent_by_child:
            previous = parent_by_child[child][1]
            return False, f'link {child} has multiple parents ({previous}, {joint_name})'
        parent_by_child[child] = (parent, joint_name)

    roots = links - set(parent_by_child)
    if len(roots) != 1:
        return False, f'expected one root link, found {len(roots)}'

    # Follow parent chains to detect cycles that a single-root check can miss.
    for start in links:
        visited = set()
        current = start
        while current in parent_by_child:
            if current in visited:
                return False, f'joint cycle involving link {current}'
            visited.add(current)
            current = parent_by_child[current][0]

    return True, ''


def load_obj_vertices(obj_path):
    """Read an OBJ file and return its (N, 3) vertex array."""
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts) if verts else np.zeros((0, 3))


def parse_urdf_visual_meshes(urdf_path):
    """Return the visual-mesh records associated with every URDF link.

    The result maps link names to filename, scale, xyz, and rpy records.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    result = {}
    for link in root.findall('.//link'):
        link_name = link.get('name')
        for visual in link.findall('visual'):
            origin = visual.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
            geom = visual.find('geometry')
            mesh_elem = geom.find('mesh') if geom is not None else None
            if mesh_elem is not None:
                filename = mesh_elem.get('filename')
                scale = [float(x) for x in mesh_elem.get('scale', '1 1 1').split()]
                if link_name not in result:
                    result[link_name] = []
                result[link_name].append({
                    'filename': filename,
                    'scale': scale,
                    'xyz': xyz,
                    'rpy': rpy
                })
    return result


def parse_urdf_link_joint_map(urdf_path):
    """Map URDF link names to PyBullet indices; -1 denotes the base link."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # The base is the first link that has no parent joint.
    child_links = set()
    all_links = set()
    for link in root.findall('.//link'):
        all_links.add(link.get('name'))
    for joint in root.findall('.//joint'):
        child = joint.find('child')
        if child is not None:
            child_links.add(urdf_link_name(child))
    base_links = all_links - child_links

    # PyBullet assigns link indices in joint declaration order.
    link_index_map = {}
    idx = 0
    for joint in root.findall('.//joint'):
        child = joint.find('child')
        if child is not None:
            link_index_map[urdf_link_name(child)] = idx
            idx += 1
    for bl in base_links:
        link_index_map[bl] = -1
    return link_index_map


def compute_visual_aabb(body_id, urdf_path):
    """Compute the true world-space visual AABB from OBJ vertices.

    This is more accurate than collision-based ``getAABB`` for visual-only
    assets.
    """
    mesh_info = parse_urdf_visual_meshes(urdf_path)
    link_index_map = parse_urdf_link_joint_map(urdf_path)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    n_j = p.getNumJoints(body_id)

    # Collect the world-frame pose of every link.
    link_poses = {}
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    link_poses[-1] = (base_pos, base_orn)
    if n_j > 0:
        link_states = p.getLinkStates(body_id, list(range(n_j)))
        for i, ls in enumerate(link_states):
            link_poses[i] = (ls[0], ls[1])

    all_min = np.array([float('inf')] * 3)
    all_max = np.array([float('-inf')] * 3)

    for link_name, meshes in mesh_info.items():
        link_idx = link_index_map.get(link_name, None)
        if link_idx is None or link_idx not in link_poses:
            continue

        world_pos, world_orn = link_poses[link_idx]

        for mi in meshes:
            # Resolve the mesh path relative to the URDF.
            mesh_path = os.path.normpath(os.path.join(urdf_dir, mi['filename']))
            if not os.path.exists(mesh_path):
                continue

            verts = load_obj_vertices(mesh_path)
            if len(verts) == 0:
                continue

            # Apply visual scale.
            scale = np.array(mi['scale'])
            verts = verts * scale

            # Apply the visual shape's local xyz and rpy offset.
            local_xyz = mi['xyz']
            local_rpy = mi['rpy']
            # Convert rpy to a rotation matrix.
            local_orn = p.getQuaternionFromEuler(local_rpy)
            # Apply the local rotation and translation to each vertex.
            transformed = []
            for v in verts:
                tv, _ = p.multiplyTransforms(local_xyz, local_orn, v.tolist(), [0, 0, 0, 1])
                transformed.append(tv)
            transformed = np.array(transformed)

            # Apply the link's world-frame transform.
            for v in transformed:
                wv, _ = p.multiplyTransforms(world_pos, world_orn, v.tolist(), [0, 0, 0, 1])
                all_min = np.minimum(all_min, np.array(wv))
                all_max = np.maximum(all_max, np.array(wv))

    if np.isinf(all_min[0]):
        # Fall back to PyBullet's collision-based AABB.
        return compute_collision_aabb(body_id)

    return all_min, all_max


def compute_collision_aabb(body_id):
    """Return PyBullet's collision AABB, which may miss visual-only geometry."""
    n_j = p.getNumJoints(body_id)
    aabb_min = np.array([float('inf')] * 3)
    aabb_max = np.array([float('-inf')] * 3)
    for link_idx in [-1] + list(range(n_j)):
        link_aabb_min, link_aabb_max = p.getAABB(body_id, link_idx)
        if link_aabb_min is None:
            continue
        aabb_min = np.minimum(aabb_min, np.array(link_aabb_min))
        aabb_max = np.maximum(aabb_max, np.array(link_aabb_max))
    return aabb_min, aabb_max


def rotate_base(body_id, euler_xyz):
    """Apply an additional rotation on top of the current base-link pose."""
    pos, orn = p.getBasePositionAndOrientation(body_id)
    rot_quat = p.getQuaternionFromEuler(euler_xyz)
    new_orn = p.multiplyTransforms([0, 0, 0], rot_quat, [0, 0, 0], orn)[1]
    p.resetBasePositionAndOrientation(body_id, pos, new_orn)


def align_urdf(body_id, urdf_path, margin=0.3):
    """Center a URDF in x/y and place it above the z=0 ground plane.

    Visual OBJ geometry, rather than collision geometry, defines the AABB.
    """
    # Step once after rotation so PyBullet updates all link poses.
    p.setGravity(0, 0, 0)
    p.stepSimulation()
    p.setGravity(0, 0, -9.81)

    aabb_min, aabb_max = compute_visual_aabb(body_id, urdf_path)

    if np.isinf(aabb_min[0]):
        return

    aabb_center = (aabb_min + aabb_max) / 2.0

    # Center x/y at the origin and place the lowest point above the ground.
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    dx = 0.0 - aabb_center[0]
    dy = 0.0 - aabb_center[1]
    dz = margin - aabb_min[2]
    new_pos = (base_pos[0] + dx, base_pos[1] + dy, base_pos[2] + dz)
    p.resetBasePositionAndOrientation(body_id, new_pos, base_orn)


# Configuration
METHODS = [
    'physxanything',
    'articulate',
    'articulate_new',
    'articulate_wild',
    'urdformer',
    'physx3d',
    'physx3d_new',
    'physx3d_wild',
    'gt',
    'gt_new',
    'urdfanything',
    'urdfanything_new',
    'urdfanything_wild',
    'urdfanything_wild_new',
]

URDF_ANYTHING_ROOT = '/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets'
URDF_ANYTHING_NEW_ROOT = '/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets_new'
URDF_ANYTHING_WILD_ROOT = '/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets_wild'
URDF_ANYTHING_WILD_NEW_ROOT = '/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets_wild_new'
ARTICULATE_ROOT = '/mnt/data/zhangzhaodong/articulate-anything/results/demo'
ARTICULATE_NEW_ROOT = '/mnt/data/zhangzhaodong/articulate-anything/results/demo_new'
ARTICULATE_WILD_ROOT = '/mnt/data/zhangzhaodong/articulate-anything/results/wild-demo'
PHYSX3D_ROOT = '/mnt/data/zhangzhaodong/PhysX-3D/outputs_demo_urdf'
PHYSX3D_NEW_ROOT = '/mnt/data/zhangzhaodong/PhysX-3D/outputs_demo_new_urdf'
PHYSX3D_WILD_ROOT = '/mnt/data/zhangzhaodong/PhysX-3D/outputs_wild_demo_urdf'
args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
overwrite = '--overwrite' in sys.argv[1:]
method_arg = args[0] if len(args) > 0 else 'physxanything'
run_methods = METHODS if method_arg == 'all' else [method_arg]

testset = {
    0: "1817",
    1: "1986",
    2: "2230",
    3: "4533",
    4: "11586",
    5: "12654",
    6: "102434",  # Red egg chair
    7: "100197",
    8: "100321",
    9: "100443",
    10: "100501",
    11: "100523",
    12: "100758",
    13: "100907",
    14: "100925",
    15: "101448",
    16: "101861"
}

def numeric_names(path):
    if not os.path.isdir(path):
        raise FileNotFoundError('asset directory does not exist: ' + path)
    return sorted([name for name in os.listdir(path) if name.isdigit()], key=lambda x: int(x))


def directory_names(path):
    """Return visible sample directories, including non-numeric wild names."""
    if not os.path.isdir(path):
        raise FileNotFoundError('asset directory does not exist: ' + path)
    names = [
        name for name in os.listdir(path)
        if not name.startswith('.') and os.path.isdir(os.path.join(path, name))
    ]
    return sorted(names, key=lambda name: (0, int(name)) if name.isdigit() else (1, name))


def numeric_image_stems(path):
    """Extract numeric IDs such as 920 from input image filenames."""
    if not os.path.isdir(path):
        raise FileNotFoundError('image directory does not exist: ' + path)
    names = []
    for filename in os.listdir(path):
        stem, extension = os.path.splitext(filename)
        if extension.lower() in ('.png', '.jpg', '.jpeg', '.webp') and stem.isdigit():
            names.append(stem)
    return sorted(set(names), key=int)


def method_names(method):
    """Discover method-native sample IDs without truncating new datasets."""
    roots = {
        'articulate': ARTICULATE_ROOT,
        'articulate_new': ARTICULATE_NEW_ROOT,
        'articulate_wild': ARTICULATE_WILD_ROOT,
        'physx3d': PHYSX3D_ROOT,
        'physx3d_new': PHYSX3D_NEW_ROOT,
        'physx3d_wild': PHYSX3D_WILD_ROOT,
        'urdfanything': URDF_ANYTHING_ROOT,
        'urdfanything_new': URDF_ANYTHING_NEW_ROOT,
        'urdfanything_wild': URDF_ANYTHING_WILD_ROOT,
        'urdfanything_wild_new': URDF_ANYTHING_WILD_NEW_ROOT,
    }
    if method in roots:
        return directory_names(roots[method])
    if method == 'gt_new':
        return numeric_image_stems('./demo_new')
    return numeric_names('./test_demo')

def method_paths(method, name):
    if method == 'physxanything':
        return os.path.join('./test_demo', name, 'basic.urdf'), './evaluation_video_physxanything'
    if method in ('articulate', 'articulate_new', 'articulate_wild'):
        roots = {
            'articulate': ARTICULATE_ROOT,
            'articulate_new': ARTICULATE_NEW_ROOT,
            'articulate_wild': ARTICULATE_WILD_ROOT,
        }
        savepaths = {
            'articulate': './evaluation_video_articulateanything',
            'articulate_new': './evaluation_video_articulateanything_new',
            'articulate_wild': './evaluation_video_articulateanything_wild',
        }
        root = roots[method]
        candidates = [
            os.path.join(root, name, 'joint_actor', 'iter_0', 'seed_0', 'mobility.urdf'),
            os.path.join(root, name, 'link_placement', 'iter_0', 'seed_0', 'mobility.urdf'),
        ]
        return candidates, savepaths[method]
    if method == 'urdformer':
        return os.path.join('/mnt/data/zhangzhaodong/urdformer/output', name + '.urdf'), './evaluation_video_urdformer'
    if method in ('physx3d', 'physx3d_new', 'physx3d_wild'):
        roots = {
            'physx3d': PHYSX3D_ROOT,
            'physx3d_new': PHYSX3D_NEW_ROOT,
            'physx3d_wild': PHYSX3D_WILD_ROOT,
        }
        savepaths = {
            'physx3d': './evaluation_video_physx3d',
            'physx3d_new': './evaluation_video_physx3d_new',
            'physx3d_wild': './evaluation_video_physx3d_wild',
        }
        return os.path.join(roots[method], name, 'urdf_export', 'mobility.urdf'), savepaths[method]
    if method == 'gt':
        gt_id = testset[int(name)]
        return os.path.join('./dataset/PhysX_mobility/urdf', gt_id + '.urdf'), './evaluation_video_gt'
    if method == 'gt_new':
        return os.path.join('./dataset/PhysX_mobility/urdf', name + '.urdf'), './evaluation_video_gt_new'
    if method.startswith('urdfanything'):
        roots = {
            'urdfanything': URDF_ANYTHING_ROOT,
            'urdfanything_new': URDF_ANYTHING_NEW_ROOT,
            'urdfanything_wild': URDF_ANYTHING_WILD_ROOT,
            'urdfanything_wild_new': URDF_ANYTHING_WILD_NEW_ROOT,
        }
        root = roots[method]
        # Prefer the original textured reconstruction and fall back for legacy assets.
        candidates = [
            os.path.join(root, name, 'mesh_reconstruction', 'mobility.urdf'),
            os.path.join(root, name, 'point_reconstruction', 'pred', 'mobility.urdf'),
        ]
        return candidates, './evaluation_video_' + method
    raise ValueError('unknown method: ' + method)

# Camera parameters shared by every method.
CAM_TARGET = [0, 0, 0.8]
CAM_DIST   = 3.5
CAM_YAW    = -45
CAM_PITCH  = -25
CAM_FOV    = 60

# Use a fresh physics connection for every object. ``resetSimulation`` does
# not clear PyBullet's process-level mesh cache, and many generated URDFs use
# identical relative mesh names such as ``./objs/0/0.obj``. Reusing one
# connection can therefore show the first object's geometry for later assets.
# A fresh connection also prevents stale TinyRenderer visual instances.

for method in run_methods:
    if method not in METHODS:
        raise ValueError('unknown method: ' + method)
    print('render method:', method)
    namelist = method_names(method)
    method_rendered = 0
    method_existing = 0
    method_unresolved = 0
    method_total = 0
    for name in namelist:
        method_total += 1
        item_done = False
        URDF_PATH, savepath = method_paths(method, name)
        URDF_PATHS = URDF_PATH if isinstance(URDF_PATH, list) else [URDF_PATH]
        for URDF_PATH in URDF_PATHS:
            os.makedirs(os.path.join(savepath), exist_ok=True)

            OUT_MP4   = os.path.abspath(os.path.join(savepath, name+'.mp4'))
            if os.path.exists(OUT_MP4) and not overwrite:
                    print(f"skip existing: {OUT_MP4}")
                    method_existing += 1
                    item_done = True
                    break
            if not os.path.exists(URDF_PATH):
                    print(f"missing urdf: {URDF_PATH}")
                    continue
            valid_urdf, invalid_reason = validate_urdf_tree(URDF_PATH)
            if not valid_urdf:
                    print(f"invalid urdf, try next if available: {URDF_PATH}: {invalid_reason}")
                    continue
            FPS       = 30
            SIM_HZ    = 240
            DURATION  = 2.0
            W, H      = 512, 512

            cid = None
            try:
                # Create a clean geometry cache and renderer for every object.
                cid = p.connect(p.DIRECT)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.setTimeStep(1.0 / SIM_HZ)

                plane = p.loadURDF("plane.urdf")
                # Load the absolute URDF path. Loading a basename after chdir
                # would resolve repeated relative OBJ names to the same cache
                # key and could reuse geometry from a previous object.
                robot = p.loadURDF(os.path.abspath(URDF_PATH), useFixedBase=True)

                # PhysX-Mobility GT is Y-up; rotate +90 degrees around x for Z-up.
                if method in ('gt', 'gt_new'):
                    rotate_base(robot, [math.pi / 2, 0, 0])

                # Rotate Articulate/URDFormer assets around z so their front
                # views match the camera convention used by the other methods.
                if method.startswith('articulate') or method == 'urdformer':
                    rotate_base(robot, [0, 0, -math.pi / 2])

                # Center and ground the URDF using its true visual-mesh AABB.
                align_urdf(robot, URDF_PATH, margin=0.3)

                n_j = p.getNumJoints(robot)
                joint_idxs = [j for j in range(n_j)
                            if p.getJointInfo(robot, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]

                # Keep camera parameters identical for all methods.
                view = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, CAM_DIST, CAM_YAW, CAM_PITCH, 0, 2)
                proj = p.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=W/float(H), nearVal=0.01, farVal=10)

                writer = imageio.get_writer(OUT_MP4, fps=FPS, quality=9)

                steps = int(DURATION * SIM_HZ)
                render_every = int(SIM_HZ // FPS)
                for t in range(steps):
                    for i, j in enumerate(joint_idxs):
                        target = 0.8 * math.sin(2 * math.pi * (t / SIM_HZ) * 0.5 + i*0.3)
                        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=target, force=500)

                    p.stepSimulation()

                    if t % render_every == 0:
                        _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
                        frame = np.uint8(rgba)[..., :3]
                        writer.append_data(frame)

                writer.close()
                print(f"save: {OUT_MP4}")
                method_rendered += 1
                item_done = True
                break
            except Exception as exc:
                print(f"failed urdf, try next if available: {URDF_PATH}: {exc}")
            finally:
                if cid is not None:
                    p.disconnect(cid)
        if not item_done:
            method_unresolved += 1
    method_complete = method_rendered + method_existing
    print(
        f"{method}: complete {method_complete}/{method_total} "
        f"(rendered now={method_rendered}, existing={method_existing}, "
        f"missing/failed={method_unresolved})"
    )
