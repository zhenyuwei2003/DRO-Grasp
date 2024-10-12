import os
import sys
import subprocess
from tqdm import tqdm
from termcolor import cprint
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.controller import controller


def validate_depth(hand, object_name, q_list_validate, threshold=0.005, exact=True):
    """
    Calculate the penetration depth of predicted grasps into the object.

    :param hand: HandModel()
    :param object_name: str
    :param q_list_validate: list, joint values to validate
    :param threshold: float, criteria for determining success in depth
    :param exact: bool, if false, use point cloud instead of mesh to compute (much faster with minor error)
    :param print_info: bool, whether to print progress information
    :return: (list<bool>, list<float>), success list & depth list
    """
    name = object_name.split('+')
    if exact:
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
        object_mesh = trimesh.load_mesh(object_path)
    else:
        object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
        object_pc_normals = torch.load(object_path).to(hand.device)
        object_pc = object_pc_normals[:, :3]
        normals = object_pc_normals[:, 3:]

    result_list = []
    depth_list = []
    q_list_initial = []
    for q in q_list_validate:
        initial_q, _ = controller(hand.robot_name, q)
        q_list_initial.append(initial_q)
    for q in tqdm(q_list_initial):
        robot_pc = hand.get_transformed_links_pc(q)[:, :3]
        if exact:
            robot_pc = robot_pc.cpu()
            _, distance, _ = trimesh.proximity.ProximityQuery(object_mesh).on_surface(robot_pc)
            distance = distance[object_mesh.contains(robot_pc)]
            depth = distance.max() if distance.size else 0.
        else:
            distance = torch.cdist(robot_pc, object_pc)
            distance, index = torch.min(distance, dim=-1)
            object_pc_indexed, normals_indexed = object_pc[index], normals[index]
            get_sign = torch.vmap(lambda x, y: torch.where(torch.dot(x, y) >= 0, 1, -1))
            signed_distance = distance * get_sign(robot_pc - object_pc_indexed, normals_indexed)
            depth, _ = torch.min(signed_distance, dim=-1)
            depth = -depth.item() if depth.item() < 0 else 0.

        result_list.append(depth <= threshold)
        depth_list.append(round(depth * 1000, 2))

    return result_list, depth_list


def validate_isaac(robot_name, object_name, q_batch, gpu: int = 0):
    """
    Wrap function for subprocess call (isaac_main.py) to avoid Isaac Gym GPU memory leak problem.

    :param robot_name: str
    :param object_name: str
    :param q_batch: torch.Tensor, joint values to validate
    :param gpu: int
    :return: (list<bool>, list<float>), success list & info list
    """
    os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
    q_file_path = str(os.path.join(ROOT_DIR, f'tmp/q_list_validate_{gpu}.pt'))
    torch.save(q_batch, q_file_path)
    batch_size = q_batch.shape[0]
    args = [
        'python',
        os.path.join(ROOT_DIR, 'validation/isaac_main.py'),
        '--mode', 'validation',
        '--robot_name', robot_name,
        '--object_name', object_name,
        '--batch_size', str(batch_size),
        '--q_file', q_file_path,
        '--gpu', str(gpu),
        # '--use_gui'
    ]
    ret = subprocess.run(args, capture_output=True, text=True)
    try:
        ret_file_path = os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{gpu}.pt')
        save_data = torch.load(ret_file_path)
        success = save_data['success']
        q_isaac = save_data['q_isaac']
        os.remove(q_file_path)
        os.remove(ret_file_path)
    except FileNotFoundError as e:
        cprint(f"Caught a ValueError: {e}", 'yellow')
        cprint(ret.stdout.strip(), 'blue')
        cprint(ret.stderr.strip(), 'red')
        exit()
    return success, q_isaac
