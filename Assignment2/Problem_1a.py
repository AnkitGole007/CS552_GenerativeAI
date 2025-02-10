import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from configs import *

class FCVAE(nn.Module):
    def __init__(self, latent_dim):
        super(FCVAE, self).__init__()
        self.latent_dim = latent_dim


        self.fc1 = nn.Linear(24 * 24, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.fc4 = nn.Linear(latent_dim, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 24 * 24)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        out = torch.sigmoid(self.fc7(h))
        out = out.view(-1, 1, 24, 24)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def fc_vae_loss(reconstructed, x, mu, logvar, beta):
    reconstruction_loss = F.binary_cross_entropy(reconstructed, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_div


def main():
    faces = np.load('faces_vae.npy')
    faces = faces.astype(np.float32) / 255.0
    faces = torch.tensor(faces).unsqueeze(1)

    dataset = TensorDataset(faces)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCVAE(LATENT_DIM_FCVAE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(x)
            loss = fc_vae_loss(reconstructed, x, mu, logvar, BETA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/fc_vae_model.pth")

    model.eval()
    with torch.no_grad():
        z = torch.randn(100, LATENT_DIM_FCVAE).to(device)
        generated_faces = model.decode(z).cpu()
        generated_faces = generated_faces.squeeze(1)

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_faces[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/fcvae_generated_faces.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
