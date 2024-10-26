import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import argparse
import trimesh
import torch
import viser

from utils.hand_model import create_hand_model
from utils.controller import controller, get_link_dir
from utils.vis_utils import vis_vector


def vis_controller_result(robot_name='shadowhand', object_name=None):
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
    metadata = torch.load(dataset_path)['metadata']
    metadata = [m for m in metadata if (object_name is None or m[1] == object_name) and m[2] == robot_name]

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    slider = server.gui.add_slider(
        label='robot',
        min=0,
        max=len(metadata) - 1,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda gui: on_update(gui.target.value))

    hand = create_hand_model(robot_name)

    def on_update(idx):
        q, object_name, _ = metadata[idx]
        outer_q, inner_q = controller(robot_name, q)

        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
        object_trimesh = trimesh.load_mesh(object_path)
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            opacity=0.75
        )

        robot_trimesh = hand.get_trimesh_q(q)["visual"]
        server.scene.add_mesh_simple(
            'origin',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.75
        )
        robot_trimesh = hand.get_trimesh_q(outer_q)["visual"]
        server.scene.add_mesh_simple(
            'outer',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(255, 149, 71),
            opacity=0.75
        )
        robot_trimesh = hand.get_trimesh_q(inner_q)["visual"]
        server.scene.add_mesh_simple(
            'inner',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(255, 111, 190),
            opacity=0.75
        )

    while True:
        time.sleep(1)


def vis_hand_direction(robot_name='shadowhand'):
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    hand = create_hand_model(robot_name, device='cpu')
    q = hand.get_canonical_q()
    joint_orders = hand.get_joint_orders()
    lower, upper = hand.pk_chain.get_joint_limits()

    canonical_trimesh = hand.get_trimesh_q(q)["visual"]
    server.scene.add_mesh_simple(
        robot_name,
        canonical_trimesh.vertices,
        canonical_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

    pk_chain = hand.pk_chain
    status = pk_chain.forward_kinematics(q)
    joint_dots = {}
    for frame_name in pk_chain.get_frame_names():
        frame = pk_chain.find_frame(frame_name)
        joint = frame.joint
        link_dir = get_link_dir(robot_name, joint.name)
        if link_dir is None:
            continue

        frame_transform = status[frame_name].get_matrix()[0]
        axis_dir = frame_transform[:3, :3] @ joint.axis
        link_dir = frame_transform[:3, :3] @ link_dir
        normal_dir = torch.cross(axis_dir, link_dir, dim=0)
        axis_origin = frame_transform[:3, 3]
        origin_dir = -axis_origin / torch.norm(axis_origin)
        joint_dots[joint.name] = float(torch.dot(normal_dir, origin_dir))

        print(joint.name, joint_orders.index(joint.name), joint_dots[joint.name])
        vec_mesh = vis_vector(
            axis_origin.numpy(),
            vector=normal_dir.numpy(),
            length=0.03,
            cyliner_r=0.001,
            color=(0, 255, 0)
        )
        server.scene.add_mesh_trimesh(joint.name, vec_mesh, visible=True)

    current_q = [0 if i < 6 else lower[i] * 0.75 + upper[i] * 0.25 for i in range(hand.dof)]

    def update(joint_idx, joint_q):
        current_q[joint_idx] = joint_q
        trimesh = hand.get_trimesh_q(torch.tensor(current_q))["visual"]
        server.scene.add_mesh_simple(
            robot_name,
            trimesh.vertices,
            trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8
        )

    for i, joint_name in enumerate(joint_orders):
        if joint_name in joint_dots.keys():
            slider = server.gui.add_slider(
                label=joint_name,
                min=round(lower[i], 2),
                max=round(upper[i], 2),
                step=(upper[i] - lower[i]) / 100,
                initial_value=current_q[i],
            )
            slider.on_update(lambda gui: update(gui.target.order - 1, gui.target.value))
        else:
            slider = server.gui.add_slider(label='<ignored_joint>', min=0, max=1, step=1, initial_value=0)

    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str)
    parser.add_argument('--controller', action='store_true')
    args = parser.parse_args()

    if args.controller:
        vis_controller_result(args.robot_name)
    else:
        vis_hand_direction(args.robot_name)
