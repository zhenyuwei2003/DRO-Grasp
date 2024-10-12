import os
import sys
import time
import argparse
import warnings
import torch
import multiprocessing as mp
import subprocess
from termcolor import cprint
from tqdm import tqdm
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def worker(robot_name, object_name, batch_size, gpu):
    args = [
        'python',
        os.path.join(ROOT_DIR, 'validation/isaac_main.py'),
        '--mode', 'filter',
        '--robot_name', robot_name,
        '--object_name', object_name,
        '--batch_size', str(batch_size),
        '--gpu', gpu
    ]
    # for arg in args:
    #     print(arg, end=' ')
    # print('\n')
    start_time = time.time()
    ret = subprocess.run(args, capture_output=True, text=True)
    end_time = time.time()
    info = ret.stdout.strip().splitlines()[-1]
    cprint(f'{info:80}', 'light_blue', end=' ')
    cprint(f'time: {end_time - start_time:.2f} s', 'yellow')
    if not info.startswith('<'):
        cprint(ret.stderr.strip(), 'red')


def filter_dataset(gpu_list):
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')
    info = torch.load(dataset_path, map_location='cpu')['info']
    if 'cmap_func' in info:
        del info['cmap_func']

    pool = mp.Pool(processes=len(gpu_list))

    gpu_index = 0
    for robot_name in info.keys():
        for object_name in info[robot_name]['num_per_object'].keys():
            batch_size = info[robot_name]['num_per_object'][object_name]
            gpu = gpu_list[gpu_index % len(gpu_list)]
            pool.apply_async(worker, args=(robot_name, object_name, batch_size, gpu))
            gpu_index += 1

    pool.close()
    pool.join()


def post_process(with_heatmap=False):
    """
    dataset = {
        'info': {
            <robot_name>: {
                'robot_name': str,
                'num_total': int,
                'num_upper_object': int,
                'num_per_object': {
                    <object_name>: int,
                    ...
                }
            },
            ...
        }
        'metadata': [(<q>, <object_name>, <robot_name>), ...]
    }
    """

    if with_heatmap:
        hands = {}
        object_pcs = {}
        object_normals = {}

    info = {}
    metadata = []
    for robot_name in ['allegro', 'barrett', 'ezgripper', 'robotiq_3finger', 'shadowhand']:
        num_total = 0
        num_upper_object = 0
        num_per_object = {}

        metadata_dir = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/{robot_name}')
        object_names = os.listdir(metadata_dir)
        for file_name in tqdm(sorted(object_names)):
            object_name, _, success_num = file_name.rpartition('_')
            success_num = int(success_num[:-3])  # remove '.pt'

            num_total += success_num
            if success_num > num_upper_object:
                num_upper_object = success_num
            num_per_object[object_name] = success_num

            q = torch.load(os.path.join(metadata_dir, file_name))
            for q_idx in range(q.shape[0]):
                if with_heatmap:  # compute heatmap use GenDexGrasp method to keep consistency
                    if robot_name in hands:
                        hand = hands[robot_name]
                    else:
                        hand = create_hand_model(robot_name)
                        hands[robot_name] = hand
                    robot_pc = hand.get_transformed_links_pc(q[q_idx])[:, :3]

                    if object_name not in object_pcs:
                        name = object_name.split('+')
                        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                        mesh = trimesh.load_mesh(object_path)
                        object_pc, face_indices = mesh.sample(2048, return_index=True)
                        object_pc = torch.tensor(object_pc, dtype=torch.float32)
                        object_normal = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)
                        object_pcs[object_name] = object_pc
                        object_normals[object_name] = object_normal
                    else:
                        object_pc = object_pcs[object_name]
                        object_normal = object_normals[object_name]

                    n_robot = robot_pc.shape[0]
                    n_object = object_pc.shape[0]

                    robot_pc = robot_pc.unsqueeze(0).repeat(n_object, 1, 1)
                    object_pc = object_pc.unsqueeze(0).repeat(n_robot, 1, 1).transpose(0, 1)
                    object_normal = object_normal.unsqueeze(0).repeat(n_robot, 1, 1).transpose(0, 1)

                    object_hand_dist = (robot_pc - object_pc).norm(dim=2)
                    object_hand_align = ((robot_pc - object_pc) * object_normal).sum(dim=2)
                    object_hand_align /= (object_hand_dist + 1e-5)

                    object_hand_align_dist = object_hand_dist * torch.exp(1 - object_hand_align)
                    contact_dist = torch.sqrt(object_hand_align_dist.min(dim=1)[0])
                    contact_value_current = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)
                    heapmap = contact_value_current.unsqueeze(-1)

                    metadata.append((heapmap, q[q_idx], object_name, robot_name))
                else:
                    metadata.append((q[q_idx], object_name, robot_name))

        info[robot_name] = {
            'robot_name': robot_name,
            'num_total': num_total,
            'num_upper_object': num_upper_object,
            'num_per_object': num_per_object
        }

    dataset = {
        'info': info,
        'metadata': metadata
    }
    if with_heatmap:
        torch.save(dataset, os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset_heatmap.pt'))
        torch.save(object_pcs, os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/object_point_clouds.pt'))
    else:
        torch.save(dataset, os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt'))

    print("Post process done!")


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list',  # input format like '--gpu_list 0,1,2,3,4,5,6,7'
                        default=['0', '1', '2', '3', '4', '5', '6', '7'],
                        type=lambda string: string.split(','))
    parser.add_argument('--print_info', action='store_true')
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--with_heatmap', action='store_true')
    args = parser.parse_args()

    assert not (args.print_info and args.post_process)
    if args.print_info:
        # dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')
        dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
        info = torch.load(dataset_path, map_location=torch.device('cpu'))['info']
        if 'cmap_func' in info:
            print(f"cmap_func: {info['cmap_func']}")
            del info['cmap_func']

        for robot_name in info.keys():
            print(f"********************************")
            print(f"robot_name: {info[robot_name]['robot_name']}")
            print(f"num_total: {info[robot_name]['num_total']}")
            print(f"num_upper_object: {info[robot_name]['num_upper_object']}")
            print(f"num_per_object: {len(info[robot_name]['num_per_object'])}")
            for object_name in sorted(info[robot_name]['num_per_object'].keys()):
                print(f"    {object_name}: {info[robot_name]['num_per_object'][object_name]}")
            print(f"********************************")
    elif args.post_process:
        post_process(args.with_heatmap)
    else:
        filter_dataset(args.gpu_list)
