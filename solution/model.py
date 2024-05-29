import lightning as pl
import torch
from torch import nn
from torchmetrics import F1Score, MeanAbsoluteError
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.transforms import v2
from transformers import AutoConfig, AutoModelForImageClassification


class EarthQuakeModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        # model config
        config = AutoConfig.from_pretrained(self.hparams["model_name"], output_hidden_states=True)
        config.num_channels = self.hparams["in_chans"]
        config.num_labels = 1   # regression head output size
        hidden_size_1 = 320     # last hidden state size (reg/class head input size)
        hidden_size_2 = 1280    # class head hidden size
        aux_size_1 = 160
        aux_size_2 = 640

        # model
        self.model = AutoModelForImageClassification.from_config(config)  # magnitude prediction
        self.centroids = nn.Parameter(torch.empty(2, hidden_size_1))      # event and no event centroids
        nn.init.xavier_uniform_(self.centroids)
        self.classification_head = nn.Sequential(                         # event probability logits
            nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size_2, 1, kernel_size=1)
        )
        aux_heads = self._aux_heads(aux_size_1, aux_size_2, hidden_size_1)
        self.aux_class, self.aux_reg, self.aux_centroid = aux_heads

        # training losses
        self.reg_mae_loss = nn.L1Loss()
        self.reg_mse_loss = nn.MSELoss()
        def classification_loss(y_c, label):
            return sigmoid_focal_loss(y_c, label.to(torch.float32), reduction='mean')
        # self.class_loss = nn.BCEWithLogitsLoss()
        self.class_loss = classification_loss
        def embedding_centroids_loss(hidden_state, labels):
            # average the hidden states spatially to create mean hidden states
            hidden_state_avg = hidden_state.mean(dim=[2, 3])
            # intra-class compactness (distance from hidden states to class centroids)
            d_no_event = (hidden_state_avg[~labels]-self.centroids[0]).norm(dim=1)
            d_event = (hidden_state_avg[labels]-self.centroids[1]).norm(dim=1)
            intra_loss = (d_no_event.sum() + d_event.sum()) / labels.shape[0]
            # inter-class separation (distance between class centroids)
            inter_loss = (self.centroids[0]-self.centroids[1]).norm()
            return intra_loss, 1/inter_loss
        self.emb_loss = embedding_centroids_loss

        # performance metrics
        self.det_metric = F1Score("multiclass", num_classes=2)
        self.mag_metric = MeanAbsoluteError()

        # input pre-processing
        self.standardize = v2.Normalize(
            mean=self.hparams["mean"], std=self.hparams["std"]
            )
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

    def _aux_heads(self, aux_size_1, aux_size_2, hidden_size_1):
        classification_head = nn.Sequential(                     # event probability logits
            nn.Conv2d(aux_size_1, aux_size_2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(aux_size_2, 1, kernel_size=1)
        )
        regression_head = nn.Sequential(                         # event probability logits
            nn.Conv2d(aux_size_1, aux_size_2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(aux_size_2, 1, kernel_size=1)
        )
        centroid_head = nn.Conv2d(aux_size_1, hidden_size_1, kernel_size=1)
        return classification_head, regression_head, centroid_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # outputs:
        # x_reg - 0 < magnitude < 10
        # x_class - event probability logits
        # last hidden state (class/reg head inputs)
        x = self.model(self.standardize(x))
        x_reg = 10 * nn.functional.sigmoid(x.logits).squeeze()
        x_class = self.classification_head(x.hidden_states[-1]).squeeze()
        aux_x_reg = 10 * nn.functional.sigmoid(self.aux_reg(x.hidden_states[-2])).squeeze()
        aux_x_class = self.aux_class(x.hidden_states[-2]).squeeze()
        return x_reg, x_class, x.hidden_states[-1], aux_x_reg, aux_x_class, x.hidden_states[-2]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=0.01)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"].to(torch.float32), batch["magnitude"])
        y_r, y_c, h_s, y_r_aux, y_c_aux, h_s_aux = self(self.train_transform(sample))

        # regression loss
        mae = self.reg_mae_loss(y_r, mag)
        mse = self.reg_mse_loss(y_r, mag)
        self.log("train_mae", mae)
        self.log("train_mse", mse)
        reg_loss = mae + self.hparams["reg_mse_w"] * mse

        # classification loss
        epsilon = 0.1
        smoothed_labels = label * (1 - epsilon) + 0.5 * epsilon
        class_loss = self.class_loss(y_c, smoothed_labels)
        self.log("train_class_loss", class_loss)
        class_loss *= self.hparams["class_loss_w"]

        # feature embeddings loss
        emb_intra_loss, emb_inter_loss = self.emb_loss(h_s, label.to(bool))
        self.log("train_emb_intra_loss", emb_intra_loss)
        self.log("train_emb_inter_loss", emb_inter_loss)
        emb_loss = self.hparams["emb_loss_w"] * (emb_intra_loss + emb_inter_loss)

        # aux losses
        mae_aux = self.reg_mae_loss(y_r_aux, mag)
        mse_aux = self.reg_mse_loss(y_r_aux, mag)
        reg_loss_aux = mae_aux + self.hparams["reg_mse_w"] * mse_aux
        class_loss_aux = self.hparams["class_loss_w"] * self.class_loss(y_c_aux, smoothed_labels)
        emb_intra_loss_aux, emb_inter_loss_aux = self.emb_loss(self.aux_centroid(h_s_aux), label.to(bool))
        emb_loss_aux = self.hparams["emb_loss_w"] * (emb_intra_loss_aux + emb_inter_loss_aux)
        total_aux_loss = self.hparams["aux_loss_w"] * (reg_loss_aux + class_loss_aux + emb_loss_aux)

        # total loss
        total_loss = reg_loss + class_loss + emb_loss + total_aux_loss
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"].to(torch.float32), batch["magnitude"])
        y_r, y_c, h_s = self(sample)[:3]

        # regression loss
        mae = self.reg_mae_loss(y_r, mag)
        mse = self.reg_mse_loss(y_r, mag)
        self.log("val_mae", mae)
        self.log("val_mse", mse)
        reg_loss = mae + self.hparams["reg_mse_w"] * mse

        # classification loss
        class_loss = self.class_loss(y_c, label)
        self.log("val_class_loss", class_loss)
        class_loss *= self.hparams["class_loss_w"]

        # feature embeddings loss
        emb_intra_loss, emb_inter_loss = self.emb_loss(h_s, label.to(bool))
        self.log("val_emb_intra_loss", emb_intra_loss)
        self.log("val_emb_inter_loss", emb_inter_loss)
        emb_loss = self.hparams["emb_loss_w"] * (emb_intra_loss + emb_inter_loss)

        # total loss
        total_loss = reg_loss + class_loss + emb_loss
        self.log("val_loss", total_loss)

        # earthquake detection metric
        det_metric = self.det_metric((y_r >= 1).to(torch.int), label)
        self.log("det_metric", det_metric)

        # magnitude prediction metric
        mag_metric = self.mag_metric(y_r, mag)
        self.log("mag_metric", mag_metric)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])
        y_r, y_c, h_s = self(sample)
        self.det_metric((y_r >= 1).to(torch.int), label)
        self.mag_metric(y_r, mag)
        self.log("val_f1", self.f1)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["image"]
        y_r, y_c = self(sample)
        return y_r
