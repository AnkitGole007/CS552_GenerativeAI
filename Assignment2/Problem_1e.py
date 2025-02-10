# Problem_1e.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Problem_1a import FCVAE

# Load and preprocess the training data
faces = np.load('faces_vae.npy').astype(np.float32) / 255.0
n_samples = faces.shape[0]
flattened_faces = faces.reshape(n_samples, -1)  # (n, 24*24)

# Train PCA for 2D visualization
pca_2d = PCA(n_components=2)
pca_2d.fit(flattened_faces)

# Train PPCA (using sklearn's PCA with 200 components)
pca_ppca = PCA(n_components=200)
pca_ppca.fit(flattened_faces)
mu_ppca = pca_ppca.mean_
components_ppca = pca_ppca.components_
n_features = flattened_faces.shape[1]

# Calculate sigma squared for PPCA
total_var = np.var(flattened_faces, ddof=1, axis=0).sum()  # Unbiased variance
sum_top200 = pca_ppca.explained_variance_.sum()
sigma_sq = (total_var - sum_top200) / (n_features - 200)

# Generate PPCA samples
np.random.seed(42)
ppca_samples = []
for _ in range(20):
    z = np.random.randn(200)
    noise = np.random.normal(0, np.sqrt(sigma_sq), n_features)
    x = mu_ppca + z @ components_ppca + noise
    ppca_samples.append(x)
ppca_samples = np.array(ppca_samples)

# Generate VAE samples
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = FCVAE().to(device)
vae_model.load_state_dict(torch.load('models/fc_vae_model.pth', map_location=device))
vae_model.eval()

with torch.no_grad():
    z = torch.randn(20, 200).to(device)
    generated = vae_model.decode(z).cpu().numpy().reshape(20, -1)

# Project samples to 2D
vae_projected = pca_2d.transform(generated)
ppca_projected = pca_2d.transform(ppca_samples)

# Plot results
plt.figure(figsize=(12, 6))

# VAE plot
plt.subplot(1, 2, 1)
plt.scatter(vae_projected[:, 0], vae_projected[:, 1], c='blue', alpha=0.6)
for i in range(20):
    plt.text(vae_projected[i, 0], vae_projected[i, 1], str(i+1), fontsize=8, ha='center')
plt.title('VAE Generated Samples')

# PPCA plot
plt.subplot(1, 2, 2)
plt.scatter(ppca_projected[:, 0], ppca_projected[:, 1], c='red', alpha=0.6)
for i in range(20):
    plt.text(ppca_projected[i, 0], ppca_projected[i, 1], str(i+1), fontsize=8, ha='center')
plt.title('PPCA Generated Samples')

plt.tight_layout()
plt.savefig("outputs/vae_vs_ppca.png", dpi=300)
plt.show()