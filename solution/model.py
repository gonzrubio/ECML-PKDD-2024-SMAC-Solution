import lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import F1Score, MeanAbsoluteError
from torchvision.transforms import v2
from transformers import AutoConfig, AutoModelForImageClassification


class EarthQuakeModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 1

        config = AutoConfig.from_pretrained(self.hparams["model_name"], output_hidden_states=True)
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
        elif self.hparams["regression_loss"] == "MAE+MSE":
            mae_weight, mse_weight = self.hparams["regression_loss_coefficients"]
            def mae_mse_loss(output, target):
                mse_loss = mse_weight * F.mse_loss(output, target)
                mae_loss = mae_weight * F.l1_loss(output, target)
                return mse_loss + mae_loss
            self.regression_loss = mae_mse_loss
        else:
            print("ERROR: Regression loss must be one of MSE, MAE, or MAE+MSE")

        # Classification head (same as the paper)
        self.classification_head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 1, kernel_size=1)
        )
        self.classification_loss = nn.BCEWithLogitsLoss()

        self.train_transform =  v2.Compose([
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.12),
            v2.RandomHorizontalFlip(p=0.12),
            v2.RandomVerticalFlip(p=0.12),
            v2.RandomPerspective(p=0.12),
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 180))], p=0.12),
            v2.RandomApply(transforms=[v2.RandomCrop(size=(256, 256))], p=0.12),
            v2.RandomApply(transforms=[v2.RandomChoice(transforms=[
                v2.RandomErasing(p=1, value='random'),
                v2.RandomErasing(p=1, value=0),
                v2.RandomErasing(p=1, value=1),
                v2.RandomErasing(p=1, value=-1)
                ])], p=0.12)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardize(x)
        x = self.model(x)
        # make sure predicted magnitude is between 0-10
        x_reg = nn.functional.sigmoid(x.logits)*10
        # classificaiton task
        x_class = self.classification_head(x.hidden_states[-1])
        return x_reg.squeeze(), x_class.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0.01
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        sample = self.train_transform(sample)
        y_r, y_c = self(sample)

        #regression and classification losses
        reg_loss = self.regression_loss(y_r, mag)
        class_loss = self.classification_loss(y_c, label.to(torch.float32))
        loss = reg_loss + self.hparams["classification_loss_coefficient"]*class_loss
        self.log("train_reg_loss", reg_loss)
        self.log("train_class_loss", class_loss)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        y_r, y_c = self(sample)

        #regression and classification losses
        reg_loss = self.regression_loss(y_r, mag)
        class_loss = self.classification_loss(y_c, label.to(torch.float32))
        loss = reg_loss + self.hparams["classification_loss_coefficient"]*class_loss
        self.log("val_reg_loss", reg_loss)
        self.log("val_class_loss", class_loss)
        self.log("val_loss", loss)

        # performance metrics
        self.f1((y_r >= 1).to(torch.int), label)
        self.log("val_f1", self.f1)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        y_r, y_c = self(sample)

        self.f1((y_r >= 1).to(torch.int), label)
        self.log("val_f1", self.f1)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["image"]
        y_r, y_c = self(sample)
        return y_r
