import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Problem_1a import sample_image
import os

NO_OF_IMAGES = 100
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 500

def prepare_dataset(images):
    images = np.array(images).astype(np.float32)
    flatten_images = images.reshape(100,-1)
    unique_images = np.unique(flatten_images, axis=0)

    unique_images = torch.tensor(unique_images, dtype=torch.float32)
    dataset = TensorDataset(unique_images)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class LSTM_Model(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits.squeeze(-1)

def train(model, dataloader, epochs=EPOCHS, lr=LR):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            images = batch[0] # size(batch size,25)

            inputs = torch.cat([torch.zeros(images.size(0),1), images[:,:-1]],dim=1).unsqueeze(-1)
            targets = images

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch:{epoch + 1}/{epochs} -> Loss:{total_loss / len(dataloader):.4f}")

def generate_image(model, num_images=NO_OF_IMAGES):
    model.eval()
    generated_images = []

    for _ in range(num_images):
        image = np.zeros(25)
        inputs = torch.zeros(1,1,1)

        for i in range(25):
            logits = model(inputs)
            prob = torch.sigmoid(logits).item()
            pixel = 1 if torch.rand(1).item() < prob else 0
            image[i] = pixel
            inputs = torch.tensor([[[pixel]]], dtype=torch.float32)

        generated_images.append(image.reshape(5,5))

    return generated_images

def plot_images(images, rows=10, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='binary')  # Display image in binary colormap
        ax.axis('off')  # Turn off axes for clarity
    plt.tight_layout()
    plt.show()

def main():
    with open('image_cpts.pkl/cpts.pkl','rb') as f:
        cpts = pickle.load(f)
    images = [sample_image(cpts) for _ in range(NO_OF_IMAGES)]
    dataloader = prepare_dataset(images)

    model = LSTM_Model()
    train(model, dataloader)

    generated_images = generate_image(model)

    plot_images(generated_images)

if __name__ == '__main__':
    main()