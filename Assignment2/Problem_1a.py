import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LATENT_DIM = 64
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 50
BETA = 1.0

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64, 3,stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.m = nn.Linear(128*3*3, LATENT_DIM)
        self.log_var = nn.Linear(128*3*3, LATENT_DIM)

        self.fc_decoder = nn.Linear(LATENT_DIM, 128*3*3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def parameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        return mean + eps*std

    def forward(self, x):
        encode = self.encoder(x)
        mean = self.m(encode)
        log_var = self.log_var(encode)

        z = self.parameterize(mean,log_var)

        decode = self.fc_decoder(z).view(-1,128,3,3)
        decode = self.decoder(decode)
        return decode, mean, log_var

def vae_loss(reconstructed, x, mean, log_var, beta):
    re_loss = F.binary_cross_entropy(reconstructed,x,reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())

    return re_loss + beta * kl_loss

def main():
    faces = np.load('faces_vae.npy')
    faces = faces.astype(np.float32) / 255.0
    faces = torch.tensor(faces).unsqueeze(1)


    dataset = TensorDataset(faces)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(faces.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)

    vae.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            reconstructed, mean, log_var = vae(x)
            loss = vae_loss(reconstructed, x, mean, log_var, beta=BETA)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}")

    torch.save(vae.state_dict(),"vae_model.pth")
    vae.eval()

    z = torch.randn(100, LATENT_DIM).to(device)
    generated_faces = vae.decoder(vae.fc_decoder(z).view(-1, 128, 3, 3)).cpu().detach().numpy()

    generated_faces = generated_faces.squeeze()

    # 10x10 Collage
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_faces[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("generated_faces.png")  # Save the image
    plt.show()

if __name__ == '__main__':
    main()