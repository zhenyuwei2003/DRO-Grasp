"""
Validation visualization results will be saved in the 'vis_info/' folder.
This code is used to visualize the saved information.
"""

import os
import sys
import time
import viser
import trimesh
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def main():
    # substitute your filename here, which should be automatically saved in vis_info/ by validation.py
    file_name = 'vis_info/3robots_epoch10.pt'
    vis_info = torch.load(os.path.join(ROOT_DIR, file_name), map_location='cpu')

    def on_update(idx):
        invalid = True
        for info in vis_info:
            if idx >= info['predict_q'].shape[0]:
                idx -= info['predict_q'].shape[0]
            else:
                invalid = False
                break
        if invalid:
            print('Invalid index!')
            return

        print(info['robot_name'], info['object_name'], idx)
        print('result:', info['success'][idx])

        object_name = info['object_name'].split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{object_name[0]}/{object_name[1]}/{object_name[1]}.stl')
        object_trimesh = trimesh.load_mesh(object_path)
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            opacity=0.8
        )

        server.scene.add_point_cloud(
            'object_pc',
            info['object_pc'][idx].numpy(),
            point_size=0.0008,
            point_shape="circle",
            colors=(255, 0, 0),
            visible=False
        )

        server.scene.add_point_cloud(
            'mlat_pc',
            info['mlat_pc'][idx].numpy(),
            point_size=0.001,
            point_shape="circle",
            colors=(0, 0, 200),
            visible=False
        )

        hand = create_hand_model(info['robot_name'])

        robot_transform_trimesh = hand.get_trimesh_se3(info['predict_transform'], idx)
        server.scene.add_mesh_trimesh('transform', robot_transform_trimesh, visible=False)

        robot_trimesh = hand.get_trimesh_q(info['predict_q'][idx])['visual']
        server.scene.add_mesh_simple(
            'robot_predict',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8,
            visible=False
        )

        robot_trimesh = hand.get_trimesh_q(info['isaac_q'][idx])['visual']
        server.scene.add_mesh_simple(
            'robot_isaac',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8
        )

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    grasp_num = 0
    for info in vis_info:
        grasp_num += info['predict_q'].shape[0]

    slider = server.gui.add_slider(
        label='grasp_idx',
        min=0,
        max=grasp_num,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda _: on_update(slider.value))

    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
