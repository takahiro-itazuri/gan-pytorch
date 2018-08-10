import os
import sys
import numpy as np
import pickle
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from model.celeba import Generator, Discriminator

# hyper params
use_gpu = torch.cuda.is_available()
batch_size = 128
learning_rate = 0.0002
latent_dims = 62
num_epochs = 50
log_dir = 'logs/celeba'
data_dir = 'data/celeba'

if not os.path.exists(log_dir):
  os.makedirs(log_dir)

if not os.path.exists(data_dir):
  os.makedirs(data_dir)

# data preparation
transform = transforms.Compose([
  transforms.CenterCrop(160),
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model
G = Generator()
D = Discriminator()
if use_gpu:
  G.cuda()
  D.cuda()

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# loss
criterion = nn.BCELoss()

# training
# train function
def train():
  D.train()
  G.train()

  y_real = Variable(torch.ones(batch_size, 1))
  y_fake = Variable(torch.zeros(batch_size, 1))
  if use_gpu:
    y_real = y_real.cuda()
    y_fake = y_fake.cuda()
  
  D_running_loss = 0
  G_running_loss = 0
  num_itrs = len(data_loader.dataset) // batch_size

  for itr, (img, _) in enumerate(data_loader):
    if img.size()[0] != batch_size:
      break
    
    # Discriminator
    x_real = Variable(img)
    z = Variable(torch.randn((batch_size, latent_dims)))
    if use_gpu:
      x_real = x_real.cuda()
      z = z.cuda()
    
    
    D_optimizer.zero_grad()

    D_real = D(x_real)
    D_real_loss = criterion(D_real, y_real)

    x_fake = G(z)
    D_fake = D(x_fake.detach())
    D_fake_loss = criterion(D_fake, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    D_running_loss += D_loss.item()

    # Generator
    z = Variable(torch.randn((batch_size, latent_dims)))
    if use_gpu:
      z = z.cuda()

    G_optimizer.zero_grad()

    x_fake = G(z)
    D_fake = D(x_fake)
    G_loss = criterion(D_fake, y_real)
    G_loss.backward()
    G_optimizer.step()
    G_running_loss += G_loss.item()

    sys.stdout.write('\repoch [{}/{}], itr: [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'.format(epoch, num_epochs, itr, num_itrs, D_loss.item(), G_loss.item()))
    sys.stdout.flush()

  D_running_loss /= num_itrs
  G_running_loss /= num_itrs

  sys.stdout.write('\repoch [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}\n'.format(epoch, num_epochs, D_running_loss, G_running_loss))
  sys.stdout.flush()

  return D_running_loss, G_running_loss

# generate function
def generate(epoch):
  G.eval()

  sample_z = Variable(torch.randn((64, latent_dims)))
  if use_gpu:
    sample_z = sample_z.cuda()

  samples = G(sample_z).data.cpu()
  save_image(samples, os.path.join(log_dir, 'epoch_{:03d}.png'.format(epoch)))

history = {}
history['D_loss'] = []
history['G_loss'] = []

for epoch in range(num_epochs):
  D_loss, G_loss = train()

  history['D_loss'].append(D_loss)
  history['G_loss'].append(G_loss)

  generate(epoch)

generate(num_epochs)

plt.plot(history['D_loss'], label='D_loss')
plt.plot(history['G_loss'], label='G_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, 'loss.png'))
  
with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
  pickle.dump(history, f)

torch.save(G.state_dict(), os.path.join(log_dir, 'G.pth'))
torch.save(D.state_dict(), os.path.join(log_dir, 'D.pth'))
