import math
import os
from src.scs import ScsNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from pytorch_lightning.plugins import DDPPlugin

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

batch_size = 128
data_path = "data"

num_gpus = torch.cuda.device_count()

from .resnet import resnet18




def dataloaders():

  normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  train_transforms = transforms.Compose(
      [
          transforms.RandomCrop(32, padding=4),
          # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),

        
          normalize
      ]
  )

  test_transforms = transforms.Compose(
      [
          transforms.ToTensor(),
          normalize
      ]
  )

  trainset = CIFAR10(root='./data', train=True,
                     download=True, transform=train_transforms)
  trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=4)


  testset = CIFAR10(root='./data', train=False,
                    download=True, transform=test_transforms)
  testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=4)


  return trainloader, testloader


def create_model():
  model = torchvision.models.resnet18(pretrained=False, num_classes=10)
  model.conv1 = nn.Conv2d(3, 64, kernel_size=(
      3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  model.maxpool = nn.Identity()
  return model


class LitResnet(LightningModule):
  def __init__(self, lr=0.05):
    super().__init__()

    self.save_hyperparameters()
    self.model = ScsNet(num_classes = 10)

  def forward(self, x):
    out = self.model(x)
    return F.log_softmax(out, dim=1)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y)
    self.log("train_loss", loss)

    self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)

    return loss

  def evaluate(self, batch, stage=None):

    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y)
    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, y)

    if stage:
      self.log(f"{stage}_loss", loss, prog_bar=True)
      self.log(f"{stage}_acc", acc, prog_bar=True)

  def validation_step(self, batch, batch_idx):
    self.evaluate(batch, "val")

  def test_step(self, batch, batch_idx):
    self.evaluate(batch, "test")

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(
        self.parameters(),
        lr=self.hparams.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    steps_per_epoch = math.ceil(50000 / (batch_size * num_gpus))
    scheduler_dict = {
        "scheduler": OneCycleLR(
            optimizer,
            pct_start=0.01,
            max_lr=self.hparams.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        "interval": "step",
    }

    return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":
  seed_everything(7)

  model = LitResnet(lr=0.001)

  trainer = Trainer(
      max_epochs=10,
      gpus=num_gpus,
      sync_batchnorm=True,
      precision=32,
      strategy=DDPPlugin(find_unused_parameters=False),
      logger=TensorBoardLogger("lightning_logs/", name="resnet"),
      callbacks=[LearningRateMonitor(logging_interval="step")],
  )

  train, test = dataloaders()
  trainer.fit(model, train_dataloader=train, val_dataloaders=test)
