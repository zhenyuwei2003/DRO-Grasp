import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from types import SimpleNamespace

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CMapDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


gpu = 0
device = torch.device(f'cuda:{gpu}')
ckpt_name = 'model_3robots'  # 'model_3robots_partial', 'model_allegro', 'model_barrett', 'model_shadowhand'
batch_size = 10


def main():
    network = create_network(
        SimpleNamespace(**{
            'emb_dim': 512,
            'latent_dim': 64,
            'pretrain': None,
            'center_pc': True,
            'block_computing': True
        }),
        mode='validate'
    ).to(device)
    network.load_state_dict(torch.load(f"ckpt/model/{ckpt_name}.pth", map_location=device))
    network.eval()
    dataloader = create_dataloader(
        SimpleNamespace(**{
            'batch_size': batch_size,
            'robot_names': ['barrett', 'allegro', 'shadowhand'],
            'debug_object_names': None,
            'object_pc_type': 'random' if ckpt_name != 'model_3robots_partial' else 'partial',
            'num_workers': 16
        }),
        is_train=False
    )

    global_robot_name = None
    hand = None
    all_success_q = []
    time_list = []
    success_num = 0
    total_num = 0
    for i, data in enumerate(dataloader):
        robot_name = data['robot_name']
        object_name = data['object_name']

        if robot_name != global_robot_name:
            if global_robot_name is not None:
                all_success_q = torch.cat(all_success_q, dim=0)
                diversity_std = torch.std(all_success_q, dim=0).mean()
                times = np.array(time_list)
                time_mean = np.mean(times)
                time_std = np.std(times)

                success_rate = success_num / total_num * 100
                cprint(f"[{global_robot_name}]", 'magenta', end=' ')
                cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
                cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
                cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')

                all_success_q = []
                time_list = []
                success_num = 0
                total_num = 0
            hand = create_hand_model(robot_name, device)
            global_robot_name = robot_name

        predict_q_list = []
        for data_idx in tqdm(range(batch_size)):
            initial_q = data['initial_q'][data_idx: data_idx + 1].to(device)
            robot_pc = data['robot_pc'][data_idx: data_idx + 1].to(device)
            object_pc = data['object_pc'][data_idx: data_idx + 1].to(device)

            with torch.no_grad():
                dro = network(robot_pc, object_pc)['dro'].detach()

            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)

            layer = create_problem(hand.pk_chain, optim_transform.keys())
            start_time = time.time()
            predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
            end_time = time.time()
            # print(f"[{data_count}/{batch_size}] Optimization time: {end_time - start_time:.4f} s")
            time_list.append(end_time - start_time)

            predict_q_list.append(predict_q)

        predict_q_batch = torch.cat(predict_q_list, dim=0)

        success, isaac_q = validate_isaac(robot_name, object_name, predict_q_batch, gpu=gpu)
        succ_num = success.sum().item() if success is not None else -1
        success_q = predict_q_batch[success]
        all_success_q.append(success_q)

        cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ')
        cprint(f"Result: {succ_num}/{batch_size}({succ_num / batch_size * 100:.2f}%)", 'green')
        success_num += succ_num
        total_num += batch_size

    all_success_q = torch.cat(all_success_q, dim=0)
    diversity_std = torch.std(all_success_q, dim=0).mean()

    times = np.array(time_list)
    time_mean = np.mean(times)
    time_std = np.std(times)

    success_rate = success_num / total_num * 100
    cprint(f"[{global_robot_name}]", 'magenta', end=' ')
    cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
    cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
    cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
