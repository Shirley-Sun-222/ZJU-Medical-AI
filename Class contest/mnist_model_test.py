import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import matplotlib

def random_read_mnist_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    # Randomly pick 10 different digits (0 through 9)
    selected_images = []
    selected_labels = []
    for digit in range(10):
        # Get all indices where the label is `digit`
        digit_indices = np.where(labels == digit)[0]
        # Randomly choose one index from those
        chosen_idx = np.random.choice(digit_indices)
        selected_images.append(X[chosen_idx])
        selected_labels.append(labels[chosen_idx])
    
    selected_images = np.array(selected_images)  # shape: (10, 784)
    return selected_images, selected_labels

class MNIST_nn_dummy:
    def predict(self, image):
        """Pretend to do digit classification."""
        time.sleep(0.2)
        return random.randint(0,9)
        
def plot_images(images, true_labels, pred_labels):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        ax.set_xlabel(f"True label: {true_labels[i]} \n Pred label: {pred_labels[i]}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":  
    # Load your CSV data
    csv_file = "mnist_test.csv"  
    selected_images, selected_labels = random_read_mnist_csv(csv_file)
    # mnist_model = MNIST_nn_dummy()
    from nn import NeuralNetwork
    mnist_model = NeuralNetwork.load_model("trained_model.pkl")
    pred_labels = []
    for img in selected_images:
        pred_labels.append(mnist_model.predict(img))
    
    if pred_labels[0] == 0:
        print(pred_labels[0], selected_labels[0])
        print("The model correctly predicted the first digit.")
    plot_images(selected_images, selected_labels, pred_labels)
        

    
