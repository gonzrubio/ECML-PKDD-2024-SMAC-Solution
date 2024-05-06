import lightning as pl
import torch
from torch import nn
from torchmetrics import F1Score, MeanAbsoluteError
from torchvision.transforms import v2, RandomErasing
from transformers import AutoConfig, AutoModelForImageClassification


class EarthQuakeModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 1

        config = AutoConfig.from_pretrained(self.hparams["model_name"])
        config.num_channels = self.hparams["in_chans"]
        config.num_labels = num_classes
        self.model = AutoModelForImageClassification.from_config(config)

        self.standardize = v2.Normalize(
            mean=self.hparams["mean"], std=self.hparams["std"]
            )

        self.f1 = F1Score("multiclass", num_classes=2)
        self.regr_metric = MeanAbsoluteError()

        if self.hparams["regression_loss"] == "MSE":
            self.regression_loss = nn.MSELoss()
        elif self.hparams["regression_loss"] == "MAE":
            self.regression_loss = nn.L1Loss()
        else:
            print("ERROR Regression loss must be one of MSE or MAE")

        self.train_transform =  v2.Compose([
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 180))], p=0.2),
            v2.RandomApply(transforms=[v2.RandomCrop(size=(256, 256))], p=0.2),
            v2.RandomApply(transforms=[v2.RandomChoice(transforms=[
                RandomErasing(p=1, value='random'),
                RandomErasing(p=1, value=0),
                RandomErasing(p=1, value=1),
                RandomErasing(p=1, value=-1)
                ])], p=0.2)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardize(x)
        x = self.model(x)
        if hasattr(x, "logits"):
            x = x.logits
        # make sure predicted magnitude is between 0-10
        x = nn.functional.sigmoid(x)*10
        return x.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0.01
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     max_lr=self.hparams["lr"],
        #     pct_start=0.1,
        #     cycle_momentum=False,
        #     div_factor=1e9,
        #     final_div_factor=1e4,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        # }
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        sample = self.train_transform(sample)
        y_r = self(sample)

        loss = self.regression_loss(y_r, mag)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        loss = self.regression_loss(y_r, mag)

        self.f1((y_r >= 1).to(torch.int), label)
        self.log("val_f1", self.f1)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        self.f1((y_r >= 1).to(torch.int), label)
        self.log("val_f1", self.f1)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["image"]
        y_r = self(sample)
        return y_r
