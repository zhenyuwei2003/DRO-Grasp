import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.se3_transform import compute_link_pose
from utils.multilateration import multilateration
from utils.func_utils import calculate_depth
from utils.pretrain_utils import dist2weight, infonce_loss, mean_order


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, network, epoch_idx):
        super().__init__()
        self.cfg = cfg
        self.network = network
        self.epoch_idx = epoch_idx

        self.lr = cfg.lr

        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        object_name = batch['object_name']
        robot_links_pc = batch['robot_links_pc']
        robot_pc_initial = batch['robot_pc_initial']
        robot_pc_target = batch['robot_pc_target']
        object_pc = batch['object_pc']
        dro_gt = batch['dro_gt']

        network_output = self.network(
            robot_pc_initial,
            object_pc,
            robot_pc_target
        )

        dro = network_output['dro']
        mu = network_output['mu']
        logvar = network_output['logvar']

        mlat_pc = multilateration(dro, object_pc)
        transforms, transformed_pc = compute_link_pose(robot_links_pc, mlat_pc)

        loss = 0.

        if self.cfg.loss_kl:
            loss_kl = - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            loss_kl = torch.sqrt(1 + loss_kl ** 2) - 1
            loss_kl = loss_kl * self.cfg.loss_kl_weight
            self.log('loss_kl', loss_kl, prog_bar=True)
            loss += loss_kl

        if self.cfg.loss_r:
            loss_r = nn.L1Loss()(dro, dro_gt)
            loss_r = loss_r * self.cfg.loss_r_weight
            self.log('loss_r', loss_r, prog_bar=True)
            loss += loss_r

        if self.cfg.loss_se3:
            transforms_gt, transformed_pc_gt = compute_link_pose(robot_links_pc, robot_pc_target)
            loss_se3 = 0.
            for idx in range(len(transforms)):  # iteration over batch
                transform = transforms[idx]
                transform_gt = transforms_gt[idx]
                loss_se3_item = 0.
                for link_name in transform:
                    rel_translation = transform[link_name][:3, 3] - transform_gt[link_name][:3, 3]
                    rel_rotation = transform[link_name][:3, :3].mT @ transform_gt[link_name][:3, :3]
                    rel_rotation_trace = torch.clamp(torch.trace(rel_rotation), -1, 3)
                    rel_angle = torch.acos((rel_rotation_trace - 1) / 2)
                    loss_se3_item += torch.mean(torch.norm(rel_translation, dim=-1) + rel_angle)
                loss_se3 += loss_se3_item / len(transform)
            loss_se3 = loss_se3 / len(transforms) * self.cfg.loss_se3_weight
            self.log('loss_se3', loss_se3, prog_bar=True)
            loss += loss_se3

        if self.cfg.loss_depth:
            loss_depth = calculate_depth(transformed_pc, object_name)
            loss_depth = loss_depth * self.cfg.loss_depth_weight
            self.log('loss_depth', loss_depth, prog_bar=True)
            loss += loss_depth

        self.log("loss", loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        """
        For unknown reasons, there is a small chance that the gradients in CVAE may become NaN during backpropagation.
        In such cases, skip the iteration.
        """
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = None

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.network.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PretrainingModule(pl.LightningModule):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

        self.lr = cfg.lr
        self.temperature = cfg.temperature

        self.epoch_idx = 0
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        robot_pc_1 = batch['robot_pc_1']
        robot_pc_2 = batch['robot_pc_2']

        robot_pc_1 = robot_pc_1 - robot_pc_1.mean(dim=1, keepdims=True)
        robot_pc_2 = robot_pc_2 - robot_pc_2.mean(dim=1, keepdims=True)

        phi_1 = self.encoder(robot_pc_1)  # (B, N, 3) -> (B, N, D)
        phi_2 = self.encoder(robot_pc_2)  # (B, N, 3) -> (B, N, D)

        weights = dist2weight(robot_pc_1, func=lambda x: torch.tanh(10 * x))
        loss, similarity = infonce_loss(
            phi_1, phi_2, weights=weights, temperature=self.temperature
        )
        mean_order_error = mean_order(similarity)

        self.log("mean_order", mean_order_error)
        self.log("loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.encoder.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
