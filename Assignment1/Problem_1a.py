import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("pickle/image_cpts.pkl/cpts.pkl", "rb") as f:
    cpts = pickle.load(f)

def sample_image(cpts):
    image = np.zeros(25, dtype=int)
    for i in range(25):
      cpt = cpts[i]
      probs = cpt[tuple(image[:i])]
      print(f"Pixel {i + 1}: Retrieved probs: {probs}")
      image[i] = np.random.choice([0, 1], p=probs)
    return image.reshape(5, 5)
def plot(images):
    # Create a 10x10 grid for the 100 images
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))

    # Iterate through the grid and display each image
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='binary')  # Show the image in binary color map
        ax.axis('off')  # Turn off the axes for clarity

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def main():
    images = [sample_image(cpts) for _ in range(100)]
    return images

if __name__ == '__main__':
    images = main()
    plot(images)