import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Problem_1a import FCVAE
from Problem_1b import FCAE

faces = np.load('faces_vae.npy').astype(np.float32) / 255.0
faces = torch.tensor(faces).unsqueeze(1)

dataset = TensorDataset(faces)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

vae_model = FCVAE()  # instantiate the model architecture
vae_model.load_state_dict(torch.load('models/fc_vae_model.pth'))
vae_model.eval()
latent_codes_vae = []
with torch.no_grad():
    for batch in dataloader:
        x = batch[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        mu, _ = vae_model.encode(x)
        latent_codes_vae.append(mu.cpu().numpy())
latent_codes_vae = np.concatenate(latent_codes_vae, axis=0)

ae_model = FCAE()
ae_model.load_state_dict(torch.load('models/fc_ae_model.pth'))
ae_model.eval()
latent_codes_ae = []
with torch.no_grad():
    for batch in dataloader:
        x = batch[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        z = ae_model.encode(x)
        latent_codes_ae.append(z.cpu().numpy())
latent_codes_ae = np.concatenate(latent_codes_ae, axis=0)

# Reduce dimensionality to 2 using PCA
pca = PCA(n_components=2)
latent_vae_2D = pca.fit_transform(latent_codes_vae)
latent_ae_2D = pca.fit_transform(latent_codes_ae)

# Plotting the latent distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(latent_vae_2D[:, 0], latent_vae_2D[:, 1], s=2, alpha=0.5)
axes[0].set_title("VAE Latent Distribution (PCA)")
axes[1].scatter(latent_ae_2D[:, 0], latent_ae_2D[:, 1], s=2, alpha=0.5)
axes[1].set_title("AE Latent Distribution (PCA)")
plt.show()
