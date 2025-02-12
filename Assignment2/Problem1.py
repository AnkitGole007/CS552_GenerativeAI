import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Part 1:
# (Variational) Auto-Encoders for Faces
class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 576)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        out = torch.sigmoid(self.fc5(h))
        out = out.view(-1, 1, 24, 24)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KL

# Auto-Encoders for Faces
class AE(nn.Module):
    def __init__(self, latent_dim=100):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_latent = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 576)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        out = torch.sigmoid(self.fc5(h))
        out = out.view(-1, 1, 24, 24)
        return out

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_data(path='faces_vae.npy', batch_size=128):
    faces = np.load(path).astype(np.float32) / 255.0  # Normalize to [0, 1]
    faces = torch.tensor(faces).unsqueeze(1)  # Shape: (N, 1, 24, 24)
    dataset = TensorDataset(faces)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return faces, dataset, dataloader


def train_model(model, dataloader, optimizer, num_epochs=50, model_type='VAE', beta=0.1, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            if model_type == 'VAE':
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar, beta)
            elif model_type == 'AE':
                recon, _ = model(x)
                loss = F.binary_cross_entropy(recon, x, reduction='mean')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model


def plot_collage(images, grid_shape, save_path=None, title=None):
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1], grid_shape[0]))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < images.shape[0]:
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    faces, dataset, train_loader = load_data(batch_size=128)

    # (1a) Train VAE model
    print("\nTraining VAE with latent_dim=200 and beta=0.1...")
    vae_model = VAE(latent_dim=100).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    vae_model = train_model(vae_model, train_loader, vae_optimizer, num_epochs=50, model_type='VAE', beta=0.1, device=device)
    torch.save(vae_model.state_dict(), "./models/fc_vae_model.pth")

    # Generate 100 faces (VAE Model)
    vae_model.eval()
    with torch.no_grad():
        z = torch.randn(100, 100).to(device)
        generated_faces_vae = vae_model.decode(z).cpu().squeeze(1)  # (100, 24, 24)
    plot_collage(generated_faces_vae.numpy(), grid_shape=(10, 10), save_path="./outputs/vae_generated_faces.png", title="VAE Generated Faces")

    # (1b) Train AE model
    print("\nTraining AE with latent_dim=200...")
    ae_model = AE(latent_dim=100).to(device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    ae_model = train_model(ae_model, train_loader, ae_optimizer, num_epochs=50, model_type='AE', device=device)
    torch.save(ae_model.state_dict(), "./models/fc_ae_model.pth")

    # Generate 100 faces (AE Model)
    ae_model.eval()
    latent_codes = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            z = ae_model.encode(x)
            latent_codes.append(z)
            if len(torch.cat(latent_codes)) >= 100:
                break
    latent_codes = torch.cat(latent_codes, dim=0)[:100]
    with torch.no_grad():
        generated_faces_ae = ae_model.decode(latent_codes).cpu().squeeze(1)
    plot_collage(generated_faces_ae.numpy(), grid_shape=(10, 10), save_path="./outputs/ae_generated_faces.png", title="AE Generated Faces")


    # (1c) Compare Q(z|x) between VAE and AE models using PCA

    vae_model.eval()
    ae_model.eval()
    vae_latents = []
    ae_latents = []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            x = batch[0].to(device)
            mu, _ = vae_model.encode(x)
            vae_latents.append(mu.cpu().numpy())
            z = ae_model.encode(x)
            ae_latents.append(z.cpu().numpy())

    vae_latents = np.concatenate(vae_latents, axis=0)
    ae_latents = np.concatenate(ae_latents, axis=0)

    pca = PCA(n_components=2)
    vae_latents_2D = pca.fit_transform(vae_latents)
    ae_latents_2D = pca.fit_transform(ae_latents)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(vae_latents_2D[:, 0], vae_latents_2D[:, 1], s=2, alpha=0.5)
    axes[0].set_title("VAE Latent Distribution (PCA)")
    axes[1].scatter(ae_latents_2D[:, 0], ae_latents_2D[:, 1], s=2, alpha=0.5)
    axes[1].set_title("AE Latent Distribution (PCA)")
    plt.tight_layout()
    plt.savefig("./outputs/latent_distribution_comparison.png", dpi=300)
    plt.show()


    # (1d) Sampling Linearly in the Latent Space (VAE)

    num_interp = 20

    z_start = torch.randn(100)
    z_end = torch.randn(100)
    interp_latents = []
    for t in np.linspace(0, 1, num_interp):
        interp_latents.append(((1 - t) * z_start + t * z_end).unsqueeze(0))
    interp_latents = torch.cat(interp_latents, dim=0).to(device)

    with torch.no_grad():
        interp_faces = vae_model.decode(interp_latents).cpu().squeeze(1)
    plot_collage(interp_faces.numpy(), grid_shape=(2, 10), save_path="./outputs/vae_interpolation.png", title="VAE Interpolation in Latent Space")

    # (1e) Comparing P(x|z) between VAE and P-PCA

    faces_flat = faces.view(faces.size(0), -1).numpy()
    mean_face = np.mean(faces_flat, axis=0)
    centered_faces = faces_flat - mean_face

    pca_model = PCA(n_components=100)
    pca_model.fit(centered_faces)

    def ppca_decode(z):
        scaled_z = z * np.sqrt(pca_model.explained_variance_)
        recon = mean_face + np.dot(scaled_z, pca_model.components_)
        recon = np.clip(recon, 0, 1)
        return recon.reshape(-1, 24, 24)

    interp_latents_np = interp_latents.cpu().numpy()
    ppca_generated = ppca_decode(interp_latents_np)

    pca_for_images = PCA(n_components=2)
    pca_for_images.fit(faces_flat)

    def project_images(imgs):
        imgs_flat = imgs.reshape(imgs.shape[0], -1)
        return pca_for_images.transform(imgs_flat)

    proj_vae = project_images(interp_faces.numpy())
    proj_ppca = project_images(ppca_generated)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (proj, title) in enumerate(zip([proj_ppca, proj_vae], ['P-PCA','VAE'])):
        ax = axes[i]
        ax.scatter(proj[:, 0], proj[:, 1], s=40, alpha=0.7)
        for idx, (x_val, y_val) in enumerate(proj):
            ax.text(x_val, y_val, str(idx + 1), color='red', fontsize=12)
        ax.set_title(f"{title} Generated Faces (PCA Projection)")
    plt.tight_layout()
    plt.savefig("./outputs/generated_faces_projection.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
