import os
import sys
import argparse
import time
import viser
import matplotlib.pyplot as plt
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.network import create_encoder_network
from data_utils.CMapDataset import CMapDataset
from utils.pretrain_utils import dist2weight, infonce_loss
from utils.hand_model import create_hand_model


def main(robot_name):
    encoder = create_encoder_network(emb_dim=512, pretrain='pretrain_3robots.pth')

    dataset = CMapDataset(
        batch_size=1,
        robot_names=[robot_name],
        is_train=True,
        debug_object_names=None
    )
    data = dataset[0]
    q_1 = data['initial_q'][0]
    q_2 = data['target_q'][0]
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

    match_idx = torch.argmax(similarity[0], dim=0)

    # offset for clearer visualization result
    offset = torch.tensor([0, 0.3, 0])
    vis_pc_1 = data['robot_pc_initial'][0]
    vis_pc_2 = data['robot_pc_target'][0] + offset
    q_2[:3] += offset

    # match_tgt = vis_pc_2[match_idx]
    # match_vec = match_tgt - vis_pc_1

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    z_values = vis_pc_1[:, 1]
    z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    cmap = plt.get_cmap('rainbow')
    initial_colors = cmap(z_normalized)[:, :3]
    target_colors = initial_colors[match_idx]

    server.scene.add_point_cloud(
        'initial pc',
        vis_pc_1[:, :3].numpy(),
        point_size=0.002,
        point_shape="circle",
        colors=initial_colors
    )

    server.scene.add_point_cloud(
        'target pc',
        vis_pc_2[:, :3].numpy(),
        point_size=0.002,
        point_shape="circle",
        colors=target_colors
    )

    hand = create_hand_model(robot_name)

    robot_trimesh = hand.get_trimesh_q(q_1)["visual"]
    server.scene.add_mesh_simple(
        'robot_initial',
        robot_trimesh.vertices,
        robot_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.2
    )

    robot_trimesh = hand.get_trimesh_q(q_2)["visual"]
    server.scene.add_mesh_simple(
        'robot_target',
        robot_trimesh.vertices,
        robot_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.2
    )

    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, default='shadowhand')
    args = parser.parse_args()

    main(args.robot_name)
