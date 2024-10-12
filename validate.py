import os
import sys
import time
import warnings
from termcolor import cprint
import hydra
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CMapDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):
    device = torch.device(f'cuda:{cfg.gpu}')
    batch_size = cfg.dataset.batch_size
    print(f"Device: {device}")
    print('Name:', cfg.name)

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)
    for validate_epoch in cfg.validate_epochs:
        print(f"************************ Validating epoch {validate_epoch} ************************")
        with open(log_file_name, 'a') as f:
            print(f"************************ Validating epoch {validate_epoch} ************************", file=f)

        network = create_network(cfg.model, mode='validate').to(device)
        network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device))
        network.eval()

        dataloader = create_dataloader(cfg.dataset, is_train=False)

        global_robot_name = None
        hand = None
        all_success_q = []
        time_list = []
        success_num = 0
        total_num = 0
        vis_info = []
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
                    with open(log_file_name, 'a') as f:
                        cprint(f"[{global_robot_name}]", 'magenta', end=' ', file=f)
                        cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ', file=f)
                        cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ', file=f)
                        cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue', file=f)

                    all_success_q = []
                    time_list = []
                    success_num = 0
                    total_num = 0
                hand = create_hand_model(robot_name, device)
                global_robot_name = robot_name

            initial_q_list = []
            predict_q_list = []
            object_pc_list = []
            mlat_pc_list = []
            transform_list = []
            data_count = 0
            while data_count != batch_size:
                split_num = min(batch_size - data_count, cfg.split_batch_size)

                initial_q = data['initial_q'][data_count : data_count + split_num].to(device)
                robot_pc = data['robot_pc'][data_count : data_count + split_num].to(device)
                object_pc = data['object_pc'][data_count : data_count + split_num].to(device)

                data_count += split_num

                with torch.no_grad():
                    dro = network(
                        robot_pc,
                        object_pc
                    )['dro'].detach()

                mlat_pc = multilateration(dro, object_pc)
                transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
                optim_transform = process_transform(hand.pk_chain, transform)

                layer = create_problem(hand.pk_chain, optim_transform.keys())
                start_time = time.time()
                predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
                end_time = time.time()
                print(f"[{data_count}/{batch_size}] Optimization time: {end_time - start_time:.4f} s")
                time_list.append(end_time - start_time)

                initial_q_list.append(initial_q)
                predict_q_list.append(predict_q)
                object_pc_list.append(object_pc)
                mlat_pc_list.append(mlat_pc)
                transform_list.append(transform)

            initial_q_batch = torch.cat(initial_q_list, dim=0)
            predict_q_batch = torch.cat(predict_q_list, dim=0)
            object_pc_batch = torch.cat(object_pc_list, dim=0)
            mlat_pc_batch = torch.cat(mlat_pc_list, dim=0)
            transform_batch = {}
            for transform in transform_list:
                for k, v in transform.items():
                    transform_batch[k] = v if k not in transform_batch else torch.cat((transform_batch[k], v), dim=0)

            success, isaac_q = validate_isaac(robot_name, object_name, predict_q_batch, gpu=cfg.gpu)
            succ_num = success.sum().item() if success is not None else -1
            success_q = predict_q_batch[success]
            all_success_q.append(success_q)

            vis_info.append({
                'robot_name': robot_name,
                'object_name': object_name,
                'initial_q': initial_q_batch,
                'predict_q': predict_q_batch,
                'object_pc': object_pc_batch,
                'mlat_pc': mlat_pc_batch,
                'predict_transform': transform_batch,
                'success': success,
                'isaac_q': isaac_q
            })

            cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ')
            cprint(f"Result: {succ_num}/{batch_size}({succ_num / batch_size * 100:.2f}%)", 'green')
            with open(log_file_name, 'a') as f:
                cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ', file=f)
                cprint(f"Result: {succ_num}/{batch_size}({succ_num / batch_size * 100:.2f}%)", 'green', file=f)
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
        with open(log_file_name, 'a') as f:
            cprint(f"[{global_robot_name}]", 'magenta', end=' ', file=f)
            cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ', file=f)
            cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ', file=f)
            cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue', file=f)

        vis_info_file = f'{cfg.name}_epoch{validate_epoch}'
        os.makedirs(os.path.join(ROOT_DIR, 'vis_info'), exist_ok=True)
        torch.save(vis_info, os.path.join(ROOT_DIR, f'vis_info/{vis_info_file}.pt'))


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
