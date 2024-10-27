import os
import sys
import json
import argparse
import warnings
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.isaac_validator import IsaacValidator  # IsaacGym must be imported before PyTorch
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch


def isaac_main(
    mode: str,
    robot_name: str,
    object_name: str,
    batch_size: int,
    q_batch: torch.Tensor = None,
    gpu: int = 0,
    use_gui: bool = False
):
    """
    For filtering dataset and validating grasps.

    :param mode: str, 'filter' or 'validation'
    :param robot_name: str
    :param object_name: str
    :param batch_size: int, number of grasps in Isaac Gym simultaneously
    :param q_batch: torch.Tensor (validation only)
    :param gpu: int, specify the GPU device used by Isaac Gym
    :param use_gui: bool, whether to visualize Isaac Gym simulation process
    :return: success: (batch_size,), bool, whether each grasp is successful in Isaac Gym;
             q_isaac: (success_num, DOF), torch.float32, successful joint values after the grasp phase
    """
    if mode == 'filter' and batch_size == 0:  # special judge for num_per_object = 0 in dataset
        return 0, None
    if use_gui:  # for unknown reason otherwise will segmentation fault :(
        gpu = 0

    data_urdf_path = os.path.join(ROOT_DIR, 'data/data_urdf')
    urdf_assets_meta = json.load(open(os.path.join(data_urdf_path, 'robot/urdf_assets_meta.json')))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    object_name_split = object_name.split('+') if object_name is not None else None
    # object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.urdf'
    object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link.urdf'

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name, 
        joint_orders=joint_orders, 
        batch_size=batch_size,
        gpu=gpu, 
        is_filter=(mode == 'filter'),
        use_gui=use_gui
    )
    print("[Isaac] IsaacValidator is created.")

    simulator.set_asset(
        robot_path=os.path.join(data_urdf_path, 'robot'),
        robot_file=robot_urdf_path[21:],  # ignore 'data/data_urdf/robot/'
        object_path=os.path.join(data_urdf_path, 'object'),
        object_file=object_urdf_path
    )
    simulator.create_envs()
    print("[Isaac] IsaacValidator preparation is done.")

    if mode == 'filter':
        dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')
        metadata = torch.load(dataset_path)['metadata']
        q_batch = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
        q_batch = torch.stack(q_batch, dim=0).to(torch.device('cpu'))
    if q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)

    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
    success, q_isaac = simulator.run_sim()
    simulator.destroy()

    return success, q_isaac


# for Python scripts subprocess call to avoid Isaac Gym GPU memory leak problem
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--q_file', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_gui', action='store_true')
    args = parser.parse_args()

    print(f'GPU: {args.gpu}')
    assert args.mode in ['filter', 'validation'], f"Unknown mode: {args.mode}!"
    q_batch = torch.load(args.q_file, map_location=f'cpu') if args.q_file is not None else None
    success, q_isaac = isaac_main(
        mode=args.mode,
        robot_name=args.robot_name,
        object_name=args.object_name,
        batch_size=args.batch_size,
        q_batch=q_batch,
        gpu=args.gpu,
        use_gui=args.use_gui
    )

    success_num = success.sum().item()
    if args.mode == 'filter':
        print(f"<{args.robot_name}/{args.object_name}> before: {args.batch_size}, after: {success_num}")
        if success_num > 0:
            q_filtered = q_isaac[success]
            save_dir = str(os.path.join(ROOT_DIR, 'data/CMapDataset_filtered', args.robot_name))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(q_filtered, os.path.join(save_dir, f'{args.object_name}_{success_num}.pt'))
    elif args.mode == 'validation':
        cprint(f"[{args.robot_name}/{args.object_name}] Result: {success_num}/{args.batch_size}", 'green')
        save_data = {
            'success': success,
            'q_isaac': q_isaac
        }
        os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
        torch.save(save_data, os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{args.gpu}.pt'))
