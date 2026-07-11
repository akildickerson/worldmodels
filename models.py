import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
  """
  Variational Auto Encoder designed to work for 3, 96, 96 images. 

  3,158,851 parameters and a compression rate of 216x. 
  (3 x 96 x 96 = 27,648) -> (27,648 / 128 = 216)
  
  Ha and Schmidhuber:
  4,348,547 parameters and a compression rate of 384x
  (3 x 64 x 64 = 12,288) -> (12,288 / 32 = 384)
  """
  def __init__(self):
    super().__init__()

    # encoder
    # B, C, W, H -> (B, 3, 96, 96)
    self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # (B, 32, 48, 48)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # (B, 64, 24, 24)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # (B, 128, 12, 12)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # (B, 256, 6, 6)


    # bottle neck
    self.mu = nn.Linear(9216, 128)
    self.logvar = nn.Linear(9216, 128)

    # decoder
    self.linear = nn.Linear(128, 9216)
    self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) 
    self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
    self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
    self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    x = x.view(-1, 9216)

    mu = self.mu(x)
    logvar = self.logvar(x)
    sigma = torch.exp(0.5*logvar)
    eps = torch.randn_like(sigma)
    z = mu + sigma*eps

    x = self.linear(z)
    x = x.reshape(-1, 256, 6, 6)

    x = F.relu(self.deconv1(x))
    x = F.relu(self.deconv2(x))
    x = F.relu(self.deconv3(x))
    x = F.sigmoid(self.deconv4(x))

    return x, z, mu, logvar
  
  # Note: reshape is generally safer than view. Reshape and view behave 
  # similarily when the tensor is contigious in memory. Reshape only 
  # makes a copy when the tensor is not contigious in memory. 

def ELBOLoss(pred, target, mu, logvar, beta=1.0):
  """
  ELBOLoss as described in Kingma & Welling 2022, Auto-Encoding Variational Bayes
  """
  # Note: Beta tries to control the scale of the KL Divergence term of the loss,
  # so it doesn't dominate the loss. Dreamerv2 addreses this problem more 
  # rigourously. 

  _batch_size = pred.shape[0]
  recon = F.mse_loss(pred, target, reduction='sum') / _batch_size
  kl = (-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))) / _batch_size

  return beta * kl + recon