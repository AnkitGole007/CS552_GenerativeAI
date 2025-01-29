import pickle
import numpy as np
from graph_viz import render_graph
import matplotlib.pyplot as plt

LEARNING_RATE = 0.002
EPOCHS = 200

def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def compute_log_likelihood(E,X,a,b):
    n = X.shape[0]
    log_likelihood = 0

    for i in range(n):
        for j in range(i+1,n):
            z = a * np.dot(X[i],X[j]) + b
            prob = sigmoid(z)
            log_likelihood += E[i,j] * np.log(prob) + (1 - E[i,j]) * np.log(1 - prob)

    return log_likelihood

def compute_gradients(E,X,a,b):
    n = X.shape[0]
    dl_da = 0
    dl_db = 0
    error = 0

    for i in range(n):
        for j in range(i + 1, n):
            z = a * np.dot(X[i],X[j]) + b
            prob = sigmoid(z)
            error = E[i,j] - prob

            dl_da += error * np.dot(X[i],X[j])
            dl_db += error

    return dl_da, dl_db

def train_model(graphs, lr = LEARNING_RATE, epochs = EPOCHS):
    np.random.seed(42)
    a = np.random.randn()
    b = np.random.randn()
    log_likelihoods = []

    for epoch in range(epochs):
        total_dlda = 0
        total_dldb = 0
        total_log_likelihood = 0


        for graph in graphs:
            X,E = graph
            log_likelihood = compute_log_likelihood(E,X,a,b)
            dlda, dldb = compute_gradients(E,X,a,b)

            total_dlda += dlda
            total_dldb += dldb
            total_log_likelihood += log_likelihood

        a += lr * total_dlda
        b += lr* total_dldb
        log_likelihoods.append(total_log_likelihood)

        if (epoch % 10) == 0:
            print(f'For Epochs: {epoch}, total Log Likelihood is:{total_log_likelihood:.4f}')

    # Plot Log-Likelihood
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), log_likelihoods, label="Log-Likelihood", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Total Log-Likelihood")
    plt.title("Log-Likelihood Progress Over Training")
    plt.legend()
    plt.grid()
    plt.show()

    return a,b

def generate_graph(a, b, num_students=15, num_features=3):
    X = np.random.uniform(-1, 1, (num_students, num_features))  # Sample feature vectors
    E = np.zeros((num_students, num_students))  # Initialize adjacency matrix

    for i in range(num_students):
        for j in range(i + 1, num_students):
            prob = sigmoid(a * np.dot(X[i], X[j]) + b)
            E[i, j] = E[j, i] = 1 if np.random.rand() < prob else 0  # Sample edge

    return {'X': X, 'E': E}

def main():
    with open('classroom_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    a, b = train_model(graphs)
    print(f"Final Parameters: a = {a:.4f}, b = {b:.4f}")

    # Generate and Visualize 5 New Graphs
    new_graphs = [generate_graph(a, b) for _ in range(5)]
    for i, graph in enumerate(new_graphs):
        print(f"Generated Graph {i + 1}")
        render_graph(graph['X'], graph['E'])  # Uses graph_viz.py to visualize

if __name__ == '__main__':
    main()







