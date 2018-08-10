import os
import numpy as np
import pickle
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model.mnist import Generator, Discriminator

use_gpu = torch.cuda.is_available()
batch_size = 1024
learning_rate = 0.0002
z_dim = 62
num_epochs = 50
sample_num = 16
log_dir = 'logs/mnist'
data_dir = 'data/mnist'

if not os.path.exists(log_dir):
  os.makedirs(log_dir)

if not os.path.exists(data_dir):
  os.makedirs(data_dir)

def train(D, G, criterion, D_optimizer, G_optimizer, data_loader):
  D.train()
  G.train()

  y_real = Variable(torch.ones(batch_size, 1))
  y_fake = Variable(torch.zeros(batch_size, 1))
  if use_gpu:
    y_real = y_real.cuda()
    y_fake = y_fake.cuda()
  
  D_running_loss = 0
  G_running_loss = 0
  for idx, (real_imgs, _) in enumerate(data_loader):
    if real_imgs.size()[0] != batch_size:
      break
    
    # Discriminator ---
    z = torch.rand((batch_size, z_dim))
    if use_gpu:
      real_imgs = Variable(real_imgs).cuda()
      z = Variable(z).cuda()
    else:
      real_imgs = Variable(real_imgs)
      z = Variable(z)
    
    D_optimizer.zero_grad()

    # real
    D_real = D(real_imgs)
    D_real_loss = criterion(D_real, y_real)

    # fake
    fake_imgs = G(z)
    D_fake = D(fake_imgs.detach())
    D_fake_loss = criterion(D_fake, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    D_running_loss += D_loss.item()

    # Generator ---
    z = torch.rand((batch_size, z_dim))
    if use_gpu:
      z = Variable(z).cuda()
    else:
      z = Variable(z)
    
    G_optimizer.zero_grad()

    fake_imgs = G(z)
    D_fake = D(fake_imgs)
    G_loss = criterion(D_fake, y_real)
    G_loss.backward()
    G_optimizer.step()
    G_running_loss += G_loss.item()

  D_running_loss /= len(data_loader)
  G_running_loss /= len(data_loader)

  return D_running_loss, G_running_loss

def generate(epoch, G, log_dir):
  G.eval()

  sample_z = torch.rand((64, z_dim))
  if use_gpu:
    sample_z = Variable(sample_z).cuda()
  else:
    sample_z = Variable(sample_z)
  
  samples = G(sample_z).data.cpu()
  save_image(samples, os.path.join(log_dir, 'epoch_{:03d}.png'.format(epoch)))


G = Generator()
D = Discriminator()
if use_gpu:
  G.cuda()
  D.cuda()

G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

history = {}
history['D_loss'] = []
history['G_loss'] = []
for epoch in range(num_epochs+1):
  D_loss, G_loss = train(D, G, criterion, D_optimizer, G_optimizer, train_loader)

  print('epoch [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'.format(epoch, num_epochs, D_loss, G_loss))
  history['D_loss'].append(D_loss)
  history['G_loss'].append(G_loss)

  if epoch % 10 == 0:
    generate(epoch, G, log_dir)

generate(num_epochs, G, log_dir)

plt.plot(history['D_loss'], label='D_loss')
plt.plot(history['G_loss'], label='G_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, 'loss.png'))

torch.save(G.state_dict(), os.path.join(log_dir, 'G.pth'))
torch.save(D.state_dict(), os.path.join(log_dir, 'D.pth'))

with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
  pickle.dump(history, f)
