# Based on https://chat.openai.com/share/936cb45f-bfa5-451f-aab4-474f4a39584f

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import AudioDataset
from Discriminator import Discriminator
from Generator import MGAN_G

# Define your PM-SQE and PASE perceptual losses.
def PM_SQE(clean, estimated):
    # TODO: Implement PM-SQE from pmsqe.py
    return torch.nn.functional.mse_loss(clean, estimated)

def PASE(clean, estimated):
    # TODO: Implement PASE from the 'pase' folder
    return torch.nn.functional.mse_loss(clean, estimated)


# TODO get files from folders
noisy_files = ["path_to_noisy1.wav", "path_to_noisy2.wav", ...]
clean_files = ["path_to_clean1.wav", "path_to_clean2.wav", ...]

dataset = AudioDataset(noisy_files, clean_files)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model instantiation
G = MGAN_G()
D = Discriminator(ndf=16)

# Device selection for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.to(device)
D.to(device)

# Hyperparameters
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_ = 1
mu = 0.25

# Optimizers
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

num_epochs = 100

# Loop over epochs
for epoch in range(num_epochs):
    for noisy_spec, clean_spec in dataloader:
        noisy_spec, clean_spec = noisy_spec.to(device), clean_spec.to(device)

        # Generate fake (estimated) speech
        fake_spec = G(noisy_spec, noisy_spec)  # assuming G's input signature
        
        ############################
        # Update Discriminator
        ###########################
        optimizerD.zero_grad()

        # First term in LD - when both inputs are clean
        ideal_hasqi = torch.ones(clean_spec.size(0), device=device)
        output = D(clean_spec, clean_spec, noisy_spec).view(-1)
        errD_real = torch.nn.functional.mse_loss(output, ideal_hasqi)
        
        # Second term in LD - when one input is estimated
        estimated_hasqi = D(clean_spec, fake_spec, noisy_spec).view(-1)
        errD_fake = torch.nn.functional.mse_loss(estimated_hasqi, ideal_hasqi)
        
        # Combine and backpropagate
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ############################
        # Update Generator
        ###########################
        optimizerG.zero_grad()

        # LG terms
        estimated_hasqi = D(clean_spec, fake_spec, noisy_spec).view(-1)
        errG_D = torch.nn.functional.mse_loss(estimated_hasqi, ideal_hasqi)
        errG_PM_SQE = lambda_ * PM_SQE(clean_spec, fake_spec)
        errG_PASE = mu * PASE(clean_spec, fake_spec)

        # Combine and backpropagate
        errG = errG_D + errG_PM_SQE + errG_PASE
        errG.backward()
        optimizerG.step()

    # Print statistics
    print(f"[{epoch+1}/{num_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
