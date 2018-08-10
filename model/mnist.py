import torch
from torch import nn

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.fc = nn.Sequential(
      nn.Linear(62, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 7 * 7),
      nn.BatchNorm1d(128 * 7 * 7),
      nn.ReLU()
    )

    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid()
    )
  
  def forward(self, z):
    x = self.fc(z)
    x = x.view(-1, 128, 7, 7)
    x = self.deconv(x)
    return x

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )

    self.fc = nn.Sequential(
      nn.Linear(128 * 7 * 7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    h = self.conv(x)
    h = h.view(-1, 128 * 7 * 7)
    h = self.fc(h)
    return h