import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl



class LitFashionMNIST(pl.LightningModule):
  def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):
    super().__init__()
    self.channels = channels
    self.width = width
    self.height = height
    self.num_classes = num_classes
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    
    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(channels * width * height, hidden_size),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(hidden_size, self.num_classes),
    )
  
  def forward(self, x):
    x = self.model(x)
    return F.log_softmax(x, dim=1)
  
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