import torch
import timm
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
import torchvision.models as models
import timm
#. sadfasdfasdf 

## still need to verify

class TimmEfficientNetClassification(pl.LightningModule):

    def __init__(self, channels, width, height, num_classes, 
                pretrain:bool=True, learning_rate=2e-4):
        
        super().__init__()
        
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.model_tag = 'efficientnet'

        self.model = timm.create_model('efficientnetv2_rw_t', pretrained=pretrain)

        ## transfer learning
        linear_size = list(self.model.children())[-1].in_features
        self.model.classifier = nn.Linear(linear_size, self.num_classes)

        # change the input channels as needed
        if self.channels != 3:
            self.model.conv_stem = nn.Conv2d(1, 24, kernel_size=7, stride=2, padding=3, bias=False)

        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
