import os
import sys
import argparse
import time
import viser
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def generate_object_pc(args):
    """ object/{contactdb, ycb}/<object_name>.pt: (num_points, 6), point xyz + normal """
    for dataset_type in ['contactdb', 'ycb']:
        input_dir = str(os.path.join(ROOT_DIR, args.object_source_path, dataset_type))
        output_dir = str(os.path.join(ROOT_DIR, args.save_path, 'object', dataset_type))
        os.makedirs(output_dir, exist_ok=True)

        for object_name in os.listdir(input_dir):
            if not os.path.isdir(os.path.join(input_dir, object_name)):  # skip json file
                continue
            print(f'Processing {dataset_type}/{object_name}...')
            mesh_path = os.path.join(input_dir, object_name, f'{object_name}.stl')
            mesh = trimesh.load_mesh(mesh_path)
            object_pc, face_indices = mesh.sample(args.num_points, return_index=True)
            object_pc = torch.tensor(object_pc, dtype=torch.float32)
            normals = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)
            object_pc_normals = torch.cat([object_pc, normals], dim=-1)
            torch.save(object_pc_normals, os.path.join(output_dir, f'{object_name}.pt'))

    print("\nGenerating object point cloud finished.")


def generate_robot_pc(args):
    output_dir = str(os.path.join(ROOT_DIR, args.save_path, 'robot'))
    output_path = str(os.path.join(output_dir, f'{args.robot_name}.pt'))
    os.makedirs(output_dir, exist_ok=True)

    hand = create_hand_model(args.robot_name, torch.device('cpu'), args.num_points)
    links_pc = hand.vertices
    sampled_pc, sampled_pc_index = hand.get_sampled_pc(num_points=args.num_points)

    filtered_links_pc = {}
    for link_index, (link_name, points) in enumerate(links_pc.items()):
        mask = [i % args.num_points for i in sampled_pc_index
                if link_index * args.num_points <= i < (link_index + 1) * args.num_points]
        links_pc[link_name] = torch.tensor(points, dtype=torch.float32)
        filtered_links_pc[link_name] = torch.tensor(points[mask], dtype=torch.float32)
        print(f"[{link_name}] original shape: {links_pc[link_name].shape}, filtered shape: {filtered_links_pc[link_name].shape}")

    data = {
        'original': links_pc,
        'filtered': filtered_links_pc
    }
    torch.save(data, output_path)
    print("\nGenerating robot point cloud finished.")

    server = viser.ViserServer(host='127.0.0.1', port=8080)
    server.scene.add_point_cloud(
        'point cloud',
        sampled_pc[:, :3].numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(0, 0, 200)
    )
    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='robot', type=str)
    parser.add_argument('--save_path', default='data/PointCloud/', type=str)
    parser.add_argument('--num_points', default=512, type=int)
    # for object pc generation
    parser.add_argument('--object_source_path', default='data/data_urdf/object', type=str)
    # for robot pc generation
    parser.add_argument('--robot_name', default='shadowhand', type=str)
    args = parser.parse_args()

    if args.type == 'robot':
        generate_robot_pc(args)
    elif args.type == 'object':
        generate_object_pc(args)
    else:
        raise NotImplementedError
