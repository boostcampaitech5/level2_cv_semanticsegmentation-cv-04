import os

import albumentations as A
import hydra
import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from models.base_module import Module
from omegaconf import DictConfig

import wandb
from dataset.data_module import DataModule
import loss 

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg["seed"])
    os.makedirs(f"./checkpoints/{cfg['exp_name']}", exist_ok=True)

    transforms = A.Compose([instantiate(aug) for _, aug in cfg["augmentation"].items()])
    datamodule = DataModule(num_workers=4, transforms=transforms)

    model = instantiate(cfg["model"]["model"])
    criterion = loss.Calc_loss()
    module = Module(model, criterion, cfg)

    logger = [WandbLogger(project="Bone Seg. Project", name=str(cfg["exp_name"]), entity="level1-cv19", config=cfg)]
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            "./checkpoints/"+cfg["exp_name"],
            "best",
            monitor="Valid Dice",
            mode="max",
            save_last=True,
            save_weights_only=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="Valid Dice", patience=5, mode="max"),
        StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start = 0.7, annealing_epochs=5)
    
    ]


    datasize = 800
    step_size = (datasize//cfg["fold"])//cfg["batch_size"]
    trainer = Trainer(max_epochs=cfg["epoch"], logger=logger, callbacks=callbacks, 
                        log_every_n_steps=step_size, 
                        check_val_every_n_epoch=1, precision="16-mixed")
    trainer.fit(module, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
