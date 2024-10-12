import os
import sys
from types import SimpleNamespace
from tqdm import tqdm
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.network import create_encoder_network
from data_utils.CMapDataset import create_dataloader
from utils.pretrain_utils import dist2weight, infonce_loss


pretrain_ckpt = "pretrain_3robots"  # name of pretrain model
robot_names = ['barrett', 'allegro', 'shadowhand']
verbose = False
data_num = 200


def main():
    encoder = create_encoder_network(emb_dim=512)

    encoder.load_state_dict(
        torch.load(
            os.path.join(ROOT_DIR, f'ckpt/pretrain/{pretrain_ckpt}.pth'),
            map_location=torch.device('cpu')
        )
    )

    for robot_name in robot_names:
        print(f"Robot: {robot_name}")
        dataloader = create_dataloader(
            SimpleNamespace(**{
                'batch_size': 1,
                'robot_names': [robot_name],
                'debug_object_names': None,
                'object_pc_type': 'random',
                'num_workers': 4
            }),
            is_train=True
        )

        orders = []
        for data_idx, data in enumerate(tqdm(dataloader, total=data_num)):
            if data_idx == data_num:
                break

            pc_1 = data['robot_pc_initial']
            pc_2 = data['robot_pc_target']

            pc_1 = pc_1 - pc_1.mean(dim=1, keepdims=True)
            pc_2 = pc_2 - pc_2.mean(dim=1, keepdims=True)

            emb_1 = encoder(pc_1).detach()
            emb_2 = encoder(pc_2).detach()

            weight = dist2weight(pc_1, func=lambda x: torch.tanh(10 * x))
            loss, similarity = infonce_loss(
                emb_1, emb_2, weights=weight, temperature=0.1
            )

            order = (similarity > similarity.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1).float().mean()
            orders.append(order)
            if verbose:
                print("\torder:", order)

        print(f"Robot: {robot_name}, Mean Order: {sum(orders) / len(orders)}\n")


if __name__ == '__main__':
    main()
