import os
import sys
import json
import math
import hydra
import random
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


class CMapDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random'
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type

        self.hands = {}
        self.dofs = []
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))

        split_json_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/split_train_validate_objects.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'] if is_train else dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects !!!")
            self.object_names = debug_object_names

        dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
        metadata = torch.load(dataset_path)['metadata']
        self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.robot_names]
        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)
        # print(len(self.metadata))
        # print(len(self.combination))

        self.object_pcs = {}
        if self.object_pc_type != 'fixed':
            for object_name in self.object_names:
                name = object_name.split('+')
                mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, _ = mesh.sample(65536, return_index=True)
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        else:
            print("!!! Using fixed object pcs !!!")

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        if self.is_train:
            robot_name_batch = []
            object_name_batch = []
            robot_links_pc_batch = []
            robot_pc_initial_batch = []
            robot_pc_target_batch = []
            object_pc_batch = []
            dro_gt_batch = []
            initial_q_batch = []
            target_q_batch = []
            for idx in range(self.batch_size):
                robot_name = random.choice(self.robot_names)
                robot_name_batch.append(robot_name)
                hand = self.hands[robot_name]
                metadata_robot = [(m[0], m[1]) for m in self.metadata if m[2] == robot_name]

                target_q, object_name = random.choice(metadata_robot)
                target_q_batch.append(target_q)
                object_name_batch.append(object_name)

                robot_links_pc_batch.append(hand.links_pc)

                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                elif self.object_pc_type == 'random':
                    indices = torch.randperm(65536)[:self.num_points]
                    object_pc = self.object_pcs[object_name][indices]
                    object_pc += torch.randn(object_pc.shape) * 0.002
                else:  # 'partial', remove 50% points
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]

                object_pc_batch.append(object_pc)

                robot_pc_target = hand.get_transformed_links_pc(target_q)[:, :3]
                robot_pc_target_batch.append(robot_pc_target)
                initial_q = hand.get_initial_q(target_q)
                initial_q_batch.append(initial_q)
                robot_pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]
                robot_pc_initial_batch.append(robot_pc_initial)

                dro = torch.cdist(robot_pc_target, object_pc, p=2)
                dro_gt_batch.append(dro)

            robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)
            robot_pc_target_batch = torch.stack(robot_pc_target_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            dro_gt_batch = torch.stack(dro_gt_batch)

            B, N = self.batch_size, self.num_points
            assert robot_pc_initial_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_initial_batch.shape}"
            assert robot_pc_target_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_target_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert dro_gt_batch.shape == (B, N, N),\
                f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): str
                'robot_links_pc': robot_links_pc_batch,  # list(len = B): dict, {link_name: (N_link, 3)}
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch,
                'object_pc': object_pc_batch,
                'dro_gt': dro_gt_batch,
                'initial_q': initial_q_batch,
                'target_q': target_q_batch
            }
        else:  # validate
            robot_name, object_name = self.combination[index]
            hand = self.hands[robot_name]

            initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
            robot_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
            object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)

            for batch_idx in range(self.batch_size):
                initial_q = hand.get_initial_q()
                robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

                if self.object_pc_type == 'partial':
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]
                else:
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]

                initial_q_batch[batch_idx] = initial_q
                robot_pc_batch[batch_idx] = robot_pc
                object_pc_batch[batch_idx] = object_pc

            B, N, DOF = self.batch_size, self.num_points, len(hand.pk_chain.get_joint_parameter_names())
            assert initial_q_batch.shape == (B, DOF), \
                f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
            assert robot_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"

            return {
                'robot_name': robot_name,  # str
                'object_name': object_name,  # str
                'initial_q': initial_q_batch,
                'robot_pc': robot_pc_batch,
                'object_pc': object_pc_batch
            }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)


def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, is_train):
    dataset = CMapDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
