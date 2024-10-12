import os
import sys
import hydra
import warnings
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_utils.CMapDataset import create_dataloader
from model.network import create_network
from model.module import TrainingModule


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg):
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    pl.seed_everything(cfg.seed)

    last_run_id = None
    last_epoch = 0
    last_ckpt_file = None
    if cfg.load_from_checkpoint:
        wandb_dir = f'output/{cfg.name}/log/{cfg.wandb.project}'
        last_run_id = os.listdir(wandb_dir)[0]
        ckpt_dir = f'{wandb_dir}/{last_run_id}/checkpoints'
        ckpt_files = os.listdir(ckpt_dir)
        for ckpt_file in ckpt_files:
            epoch = int(ckpt_file.split('-')[0].split('=')[1])
            if epoch > last_epoch:
                last_epoch = epoch
                last_ckpt_file = os.path.join(ckpt_dir, ckpt_file)
        print("***************************************************")
        print(f"Loading checkpoint from run_id({last_run_id}): epoch {last_epoch}")
        print("***************************************************")

    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        id=last_run_id,
        project=cfg.wandb.project
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true' if (cfg.model.pretrain is not None) else 'auto',
        devices=cfg.gpu,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=0.1
    )

    dataloader = create_dataloader(cfg.dataset, is_train=True)

    network = create_network(cfg.model, mode='train')
    model = TrainingModule(
        cfg=cfg.training,
        network=network,
        epoch_idx=last_epoch
    )
    model.train()

    trainer.fit(model, dataloader, ckpt_path=last_ckpt_file)
    torch.save(model.network.state_dict(), f'{cfg.training.save_dir}/epoch_{cfg.training.max_epochs}.pth')


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
