import torch
import numpy as np
import matplotlib.pyplot as plt
from Problem_1a import FCVAE
from configs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = FCVAE(LATENT_DIM_FCVAE).to(device)
vae.load_state_dict(torch.load('models/fc_vae_model.pth', map_location=device))

def latent_space(vae, num_steps=20):

    z1 = torch.randn(1, LATENT_DIM_FCVAE).to(device)
    z2 = torch.randn(1, LATENT_DIM_FCVAE).to(device)

    z = torch.stack([z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, num_steps)]).to(device)

    with torch.no_grad():
        faces = vae.decode(z).cpu().squeeze(1)

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))  # Adjusted figure size
    axes = axes.flatten()  # Flatten for easy iteration

    for i, ax in enumerate(axes):
        ax.imshow(faces[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/latent.png")
    plt.show()

latent_space(vae)