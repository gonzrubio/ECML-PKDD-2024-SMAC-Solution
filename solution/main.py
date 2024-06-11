import time

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model import EarthQuakeModel
from omegaconf import DictConfig
from torchgeo.datamodules import QuakeSetDataModule


@hydra.main(config_path="configs", config_name="mobilenet", version_base=None)
def main(args: DictConfig):

    pl.seed_everything(100195, workers=True)
    torch.set_float32_matmul_precision("medium")

    model = EarthQuakeModel(**args.model)
    data_module = QuakeSetDataModule(**args.dataset)
    data_module.train_aug = None

    run_id = time.strftime('%Y%m%d-%H%M%S')
    wandb_logger = WandbLogger(
        project="smac",
        name=run_id,
        log_model="all"
        )
    wandb_logger.watch(model, log="gradients")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        dirpath=f"checkpoints/{run_id}",
        filename="earthquake-detection-{epoch:02d}-{val_mae:.4f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        **args.trainer,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=50,
        precision="32-true",
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value"
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
