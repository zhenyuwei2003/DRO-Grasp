import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import warnings
import time
import random
import argparse
import viser
import torch

from utils.hand_model import create_hand_model
from utils.optimization import *
from utils.se3_transform import compute_link_pose


def main(robot_name):
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
    metadata = torch.load(dataset_path, map_location=torch.device('cpu'))['metadata']
    metadata = [m for m in metadata if m[2] == robot_name]
    q = random.choice(metadata)[0]

    hand = create_hand_model(robot_name, device='cpu')
    initial_q = hand.get_initial_q(q)
    pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]
    pc_target = hand.get_transformed_links_pc(q)[:, :3]
    
    transform, _ = compute_link_pose(hand.links_pc, pc_target.unsqueeze(0), is_train=False)
    optim_transform = process_transform(hand.pk_chain, transform)
    layer = create_problem(hand.pk_chain, optim_transform.keys())
    predict_q = optimization(hand.pk_chain, layer, initial_q.unsqueeze(0), optim_transform)[0]
    pc_optimize = hand.get_transformed_links_pc(predict_q)[:, :3]

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    server.scene.add_point_cloud(
        'pc_initial',
        pc_initial.numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(102, 192, 255),
        visible=False
    )

    server.scene.add_point_cloud(
        'pc_optimize',
        pc_optimize.numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(0, 0, 200)
    )

    server.scene.add_point_cloud(
        'pc_target',
        pc_target.numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(200, 0, 0)
    )

    while True:
        time.sleep(1)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str)
    args = parser.parse_args()

    main(args.robot_name)
