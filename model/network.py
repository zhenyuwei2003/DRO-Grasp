import os
import sys
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.encoder import Encoder, CvaeEncoder
from model.transformer import Transformer
from model.latent_encoder import LatentEncoder
from model.mlp import MLPKernel


def create_encoder_network(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = Encoder(emb_dim=emb_dim)
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
                map_location=device
            )
        )
    return encoder


class Network(nn.Module):
    def __init__(self, cfg, mode):
        super(Network, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)

    def forward(self, robot_pc, object_pc, target_pc=None):
        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        if self.cfg.pretrain is not None:
            robot_embedding = robot_embedding.detach()

        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding, object_embedding)
        transformer_object_outputs = self.transformer_object(object_embedding, robot_embedding)
        robot_embedding_tf = robot_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding + transformer_object_outputs["src_embedding"]

        # CVAE encoder
        if self.mode == 'train':
            grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            grasp_emb = torch.cat([robot_embedding_tf, object_embedding_tf], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)
        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)
        z = z.unsqueeze(dim=1).repeat(1, robot_embedding_tf.shape[1], 1)  # (B, N, latent_dim)

        Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim)
        Phi_B = torch.cat([object_embedding_tf, z], dim=-1)  # (B, N, emb_dim + latent_dim)

        # Compute D(R,O) matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            dro = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    dro[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_A.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_A.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_B.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_B.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
        }
        return outputs


def create_network(cfg, mode):
    network = Network(
        cfg=cfg,
        mode=mode
    )
    return network
