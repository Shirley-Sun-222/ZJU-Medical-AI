import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#——————————————————————————————核心激活函数与网络构建——————————————————————————

# ———————————————————————————— 1. 激活函数 ——————————————————————————
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def linear(z): return z
def linear_derivative(z): return np.ones_like(z)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def softmax_derivative(z): return None # 梯度通过交叉熵结合直接计算

# ———————————————————————————— 2. 全连接层 ——————————————————————————
class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, activation='relu', dim_output=False):
        self.input_dim, self.output_dim, self.dim_output = input_dim, output_dim, dim_output
        self.Z, self.A_in, self.A_out = None, None, None
        
        # He initialization
        self.W = np.random.randn(input_dim, output_dim) * (1.0 / np.sqrt(input_dim))
        self.b = np.zeros((1, output_dim))
       
        activations = {'relu': (relu, relu_derivative), 'sigmoid': (sigmoid, sigmoid_derivative),
                       'softmax': (softmax, softmax_derivative), 'linear': (linear, linear_derivative)}
        self.activation, self.activation_deriv = activations.get(activation, activations['linear'])

    def forward(self, A_in):
        self.A_in = A_in
        self.Z = self.A_in @ self.W + self.b
        self.A_out = self.activation(self.Z)
        self.print_dims("Forward pass", self.Z, self.A_in, self.A_out)
        return self.A_out

    def backward(self, dA_out, y_true=None, is_output_layer=False):
        if is_output_layer:
            dZ = self.A_out - y_true.reshape(self.A_out.shape)
        else:
            dZ = dA_out * self.activation_deriv(self.Z)
        
        N = dZ.shape[0]
        dW = (self.A_in.T @ dZ) / N
        db = np.sum(dZ, axis=0, keepdims=True) / N
        dA_in = dZ @ self.W.T
        
        self.print_dims("Backward pass", dZ, dW, db, dA_in)
        return dA_in, dW, db
        
    def print_dims(self, label, *args):
        if self.dim_output:
            print(f"{label}: ", " ".join(str(a.shape) for a in args))

# ———————————————————————————— 3. 神经网络类 ——————————————————————————
class NeuralNetwork:
    def __init__(self, layers_config):
        self.layers = [FullyConnectedLayer(in_dim, out_dim, act) for in_dim, out_dim, act in layers_config]
    
    def forward(self, X):
        A = X
        for layer in self.layers: A = layer.forward(A)
        return A
    
    def backward(self, dA, y_true):
        grads = []
        for i, layer in reversed(list(enumerate(self.layers))):
            dA, dW, db = layer.backward(dA_out=dA, y_true=y_true, is_output_layer=(i == len(self.layers)-1))
            grads.append((dW, db))
        grads.reverse()
        return grads
    
    def update_params(self, grads, lr=0.001):
        for layer, (dW, db) in zip(self.layers, grads):
            layer.W -= lr * dW; layer.b -= lr * db

    def save_model(self, file_path):
        model_data = {
            "layers_config": [(l.W.shape[0], l.W.shape[1], l.activation.__name__) for l in self.layers],
            "weights": [l.W for l in self.layers], "biases": [l.b for l in self.layers]
        }
        with open(file_path, 'wb') as f: pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f: model_data = pickle.load(f)
        nn = NeuralNetwork(model_data["layers_config"])
        for layer, W, b in zip(nn.layers, model_data["weights"], model_data["biases"]):
            layer.W, layer.b = W, b
        print(f"Model loaded from {file_path}")
        return nn

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        is_single_sample = (X.ndim == 1)

        if is_single_sample:
            X = X.reshape(1, -1)

        y_pred = self.forward(X)

        if y_pred.shape[1] == 1:
            pred = (y_pred > 0.5).astype(int).flatten()
        else:
            pred = np.argmax(y_pred, axis=1)

        return int(pred[0]) if is_single_sample else pred
    
#——————————————————————————————数据处理与评估指标————————————————————————————
# ———————————————————————————— 1. 数据读取与通用划分逻辑 ——————————————————————————
def read_mnist_csv(csv_file, target_labels=None):
    df = pd.read_csv(csv_file, header=None)
    labels, X = df.iloc[:, 0].values, df.iloc[:, 1:].values.astype(np.float32) / 255.0

    if target_labels is not None:
        mask = np.isin(labels, target_labels)
        X, labels = X[mask], labels[mask]
    
    y = np.eye(10)[labels] # One-hot
    return X, y, labels

def _shuffle_and_split(X, y, labels, train_ratio, data_use_ratio):
    indices = np.random.permutation(len(X))
    X, y, labels = X[indices], y[indices], labels[indices]
    
    use_num = int(len(X) * data_use_ratio)
    train_end = int(train_ratio * use_num)
    
    print(f"Train X: {X[:train_end].shape} | Train Y: {y[:train_end].shape}")
    print(f"Test X:  {X[train_end:use_num].shape} | Test Y:  {y[train_end:use_num].shape}")
    print(f"Unique labels: {np.unique(labels)} | Total samples: {len(labels)} | Classes: {len(np.unique(labels))}")
    return X[:train_end], y[:train_end], X[train_end:use_num], y[train_end:use_num]

def prepare_binary_data(csv_file, target_labels, train_ratio=0.8, data_use_ratio=1.0):
    X, _, labels = read_mnist_csv(csv_file, target_labels=target_labels)
    y = (labels == target_labels[1]).astype(int).reshape(-1, 1)
    return _shuffle_and_split(X, y, labels, train_ratio, data_use_ratio)

def prepare_multi_data(csv_file, train_ratio=0.8, data_use_ratio=1.0):
    X, y, labels = read_mnist_csv(csv_file)
    return _shuffle_and_split(X, y, labels, train_ratio, data_use_ratio)

# ———————————————————————————— 2. 损失函数与准确率 ——————————————————————————
def binary_cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_binary_accuracy(y_pred, y_true):
    return np.mean((y_pred > 0.5).astype(int).flatten() == y_true.flatten())

def softmax_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def compute_multi_accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# ———————————————————————————————训练与可视化——————————————————————————————————
# ———————————————————————————— 1. 训练核心模块封装 ——————————————————————————
def _train_minibatch_core(model, X_train, y_train, lr, max_epoch, batch_size, loss_fn, acc_fn):
    for layer in model.layers: layer.dim_output = False
    loss_history = []
    n_samples = X_train.shape[0]
    
    for epoch in range(max_epoch):
        indices = np.random.permutation(n_samples)
        X_shuf, y_shuf = X_train[indices], y_train[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch, y_batch = X_shuf[i:i+batch_size], y_shuf[i:i+batch_size]
            grads = model.backward(dA=model.forward(X_batch), y_true=y_batch)
            model.update_params(grads, lr=lr)
            
        y_pred_full = model.forward(X_train)
        current_loss = loss_fn(y_pred_full, y_train)
        loss_history.append(current_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{max_epoch} | Loss: {current_loss:.4f} | Train Acc: {acc_fn(y_pred_full, y_train)*100:.2f}%")
            
    return loss_history

def train_network_minibatch(model, X_train, y_train, lr=0.01, max_epoch=50, batch_size=64):
    return _train_minibatch_core(model, X_train, y_train, lr, max_epoch, batch_size, binary_cross_entropy_loss, compute_binary_accuracy)

def train_network_multiclass(model, X_train, y_train, lr=0.01, max_epoch=30, batch_size=64):
    return _train_minibatch_core(model, X_train, y_train, lr, max_epoch, batch_size, softmax_loss, compute_multi_accuracy)

# ———————————————————————————— 2. 可视化模块 ——————————————————————————
def draw_prediction_examples(model, X_test, y_test, target_labels):
    y_pred = (model.forward(X_test) > 0.5).astype(int).flatten()
    y_true = y_test.flatten()
    
    def plot_samples(indices, title):
        if len(indices) == 0: return
        fig, axes = plt.subplots(1, min(3, len(indices)), figsize=(9, 3))
        if len(indices) == 1: axes = [axes]
        for i, idx in enumerate(indices[:3]):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            t_digit = target_labels[1] if y_true[idx] == 1 else target_labels[0]
            p_digit = target_labels[1] if y_pred[idx] == 1 else target_labels[0]
            axes[i].set_title(f"True: {t_digit} | Pred: {p_digit}"); axes[i].axis('off')
        plt.suptitle(title); plt.show()

    plot_samples(np.where(y_pred == y_true)[0], "Correct Predictions")
    plot_samples(np.where(y_pred != y_true)[0], "Incorrect Predictions")

def draw_multiclass_prediction_examples(model, X_test, y_test, seed=42):
    np.random.seed(seed)
    y_pred, y_true = np.argmax(model.forward(X_test), axis=1), np.argmax(y_test, axis=1)
    
    def plot_grid(indices, title, color):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flatten()):
            class_indices = indices[y_true[indices] == i]
            if len(class_indices) > 0:
                idx = np.random.choice(class_indices)
                ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                ax.set_title(f"True: {i} | Pred: {y_pred[idx]}", color=color)
            ax.axis('off')
        plt.suptitle(title, fontsize=14); plt.tight_layout(); plt.show()

    plot_grid(np.where(y_pred == y_true)[0], "Correct Predictions (One per Digit)", "green")
    plot_grid(np.where(y_pred != y_true)[0], "Incorrect Predictions (One per Digit)", "red")

if __name__ == '__main__':
    
    #————————————————————————十分多分类、预测可视化——————————————————
    csv_file = "./mnist.csv"
    print("\n========== Task 1: 10-Class MNIST Classification ==========")
    train_x_multi, train_y_multi, test_x_multi, test_y_multi = prepare_multi_data(csv_file)

    layers_config_multi = [(784, 128, 'relu'), (128, 64, 'relu'), (64, 10, 'softmax')]
    model_multi = NeuralNetwork(layers_config_multi)

    t0 = time.time()
    train_network_multiclass(model_multi, train_x_multi, train_y_multi, lr=0.05, max_epoch=40, batch_size=64)
    print(f"---> 10 分类模型训练完毕！耗时: {time.time() - t0:.2f}s")

    test_acc_multi = compute_multi_accuracy(model_multi.forward(test_x_multi), test_y_multi)
    print(f"---> 最终测试集准确率: {test_acc_multi*100:.2f}%\n")

    print(">>> Visualizing Multi-Class Results...")
    draw_multiclass_prediction_examples(model_multi, test_x_multi, test_y_multi)

    #——————————————————————-模型保存加载——————————————————————————
    print("\n----------- 模型保存与加载验证 -------------")
    model_save_path = "mnist_trained_model.pkl"
    model_multi.save_model(model_save_path)

    model_path = "mnist_trained_model.pkl"
    load_nn = NeuralNetwork.load_model(model_path)
    loaded_acc = compute_multi_accuracy(load_nn.forward(test_x_multi), test_y_multi)
    print(f"加载后的模型在测试集上的准确率为: {loaded_acc*100:.2f}%")