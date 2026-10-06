import os
import sys
import warnings
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.module import PretrainingModule
from model.network import create_encoder_network
from data_utils.PretrainDataset import create_dataloader
from utils.device_utils import empty_device_cache, lightning_device_config, resolve_device


@hydra.main(version_base="1.2", config_path="configs", config_name="pretrain")
def main(cfg):
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    pl.seed_everything(cfg.seed)

    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        project=cfg.wandb.project
    )
    device_indices = list(cfg.get('device_indices', cfg.get('gpu', [0])))
    accelerator, devices = lightning_device_config(cfg.get('device', 'auto'), device_indices)
    print(f"Pretraining accelerator: {accelerator}; devices: {devices}")
    trainer = pl.Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.max_epochs
    )

    dataloader = create_dataloader(cfg.dataset)
    encoder = create_encoder_network(cfg.model.emb_dim)
    model = PretrainingModule(
        cfg=cfg.training,
        encoder=encoder
    )
    model.train()

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    empty_device_cache(resolve_device("auto"))
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
