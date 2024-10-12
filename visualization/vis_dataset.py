import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import trimesh
import torch
import viser
from utils.hand_model import create_hand_model

filtered = True

robot_names = ['allegro', 'barrett', 'ezgripper', 'robotiq_3finger', 'shadowhand']
object_names = [
    'contactdb+alarm_clock', 'contactdb+apple', 'contactdb+banana', 'contactdb+binoculars', 'contactdb+camera',
    'contactdb+cell_phone', 'contactdb+cube_large', 'contactdb+cube_medium', 'contactdb+cube_small',
    'contactdb+cylinder_large', 'contactdb+cylinder_medium', 'contactdb+cylinder_small', 'contactdb+door_knob',
    'contactdb+elephant', 'contactdb+flashlight', 'contactdb+hammer', 'contactdb+light_bulb', 'contactdb+mouse',
    'contactdb+piggy_bank', 'contactdb+ps_controller', 'contactdb+pyramid_large', 'contactdb+pyramid_medium',
    'contactdb+pyramid_small', 'contactdb+rubber_duck', 'contactdb+stanford_bunny', 'contactdb+stapler',
    'contactdb+toothpaste', 'contactdb+torus_large', 'contactdb+torus_medium', 'contactdb+torus_small',
    'contactdb+train', 'contactdb+water_bottle', 'ycb+baseball', 'ycb+bleach_cleanser', 'ycb+cracker_box',
    'ycb+foam_brick', 'ycb+gelatin_box', 'ycb+hammer', 'ycb+lemon', 'ycb+master_chef_can', 'ycb+mini_soccer_ball',
    'ycb+mustard_bottle', 'ycb+orange', 'ycb+peach', 'ycb+pear', 'ycb+pitcher_base', 'ycb+plum', 'ycb+potted_meat_can',
    'ycb+power_drill', 'ycb+pudding_box', 'ycb+rubiks_cube', 'ycb+sponge', 'ycb+strawberry', 'ycb+sugar_box',
    'ycb+tomato_soup_can', 'ycb+toy_airplane', 'ycb+tuna_fish_can', 'ycb+wood_block'
]

if filtered:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
else:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')
metadata = torch.load(dataset_path, map_location=torch.device('cpu'))['metadata']

def on_update(robot_idx, object_idx, grasp_idx):
    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]
    if filtered:
        metadata_curr = [m[0] for m in metadata if m[1] == object_name and m[2] == robot_name]
    else:
        metadata_curr = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    q = metadata_curr[grasp_idx % len(metadata_curr)]
    print(f"joint values: {q}")

    name = object_name.split('+')
    object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')  # visual mesh
    # object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/coacd_allinone.obj')  # collision mesh
    object_trimesh = trimesh.load_mesh(object_path)
    server.scene.add_mesh_simple(
        'object',
        object_trimesh.vertices,
        object_trimesh.faces,
        color=(239, 132, 167),
        opacity=1
    )

    hand = create_hand_model(robot_name)
    robot_trimesh = hand.get_trimesh_q(q)["visual"]
    server.scene.add_mesh_simple(
        'robot',
        robot_trimesh.vertices,
        robot_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

server = viser.ViserServer(host='127.0.0.1', port=8080)

robot_slider = server.gui.add_slider(
    label='robot',
    min=0,
    max=len(robot_names) - 1,
    step=1,
    initial_value=0
)
object_slider = server.gui.add_slider(
    label='object',
    min=0,
    max=len(object_names) - 1,
    step=1,
    initial_value=0
)
grasp_slider = server.gui.add_slider(
    label='grasp',
    min=0,
    max=199,
    step=1,
    initial_value=0
)
robot_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))
object_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))
grasp_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))

while True:
    time.sleep(1)
