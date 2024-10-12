"""
Visualize hand links to remove abundant links in removed_links.json.
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import argparse
import viser
from utils.hand_model import create_hand_model

parser = argparse.ArgumentParser()
parser.add_argument('--robot_name', type=str, default='shadowhand')
args = parser.parse_args()
robot_name = args.robot_name

hand = create_hand_model(robot_name)
meshes = hand.get_trimesh_q(hand.get_canonical_q())['parts']

server = viser.ViserServer(host='127.0.0.1', port=8080)

for name, mesh in meshes.items():
    server.scene.add_mesh_simple(
        name,
        mesh.vertices,
        mesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

while True:
    time.sleep(1)
