import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split

SOLVENT_COLS = ["H", "EA", "DCM", "MeOH", "Et2O"]
DEFAULT_CONFIG = {
    "fp_type": "rdkit",
    "fp_size": 256,
    "hidden_dims": [128, 64],
    "output_activation": "linear",
    "learning_rate": 0.0015,
    "max_epoch": 200,
    "batch_size": 32,
    "test_size": 0.2,
    "random_state": 42,
}


def _resolve_dataset_path(dataset_path=None):
    """Resolve the TLC dataset path from user input or known default locations."""
    base_dir = Path(__file__).resolve().parent
    if dataset_path is not None:
        candidate = Path(dataset_path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found: {candidate}")

    candidates = [
        base_dir / "TLC_dataset.xlsx",
        base_dir / "P2code" / "TLC_dataset.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find TLC_dataset.xlsx. Expected one of: "
        f"{candidates[0]} or {candidates[1]}"
    )


def _resolve_model_path(model_path):
    """Resolve model file path relative to this script when needed."""
    base_dir = Path(__file__).resolve().parent
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def load_clean_data(dataset_path=None):
    """Load TLC dataset and keep rows with valid SMILES and required columns."""
    path = _resolve_dataset_path(dataset_path)
    df = pd.read_excel(path)

    required_cols = ["COMPOUND_SMILES", "Rf", *SOLVENT_COLS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df = df.dropna(subset=required_cols)
    valid_smiles_mask = (
        df["COMPOUND_SMILES"]
        .astype(str)
        .apply(lambda s: Chem.MolFromSmiles(s) is not None)
    )
    return df.loc[valid_smiles_mask, required_cols].copy()


def smiles_to_fingerprint(
    smiles,
    fp_type="rdkit",
    radius=3,
    fp_size=128,
):
    """Convert SMILES to molecular fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fp_size, dtype=float)

    if fp_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
    elif fp_type == "rdkit":
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
    elif fp_type == "atompair":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)
    elif fp_type == "topological":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=fp_size
        )
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")

    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp_size,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def prepare_dataset(
    df,
    fp_type="rdkit",
    fp_size=128,
    test_size=0.2,
    random_state=42,
):
    """Build model features/targets and split into train/test sets."""
    x_list = []
    y_list = []

    for _, row in df.iterrows():
        fp_array = smiles_to_fingerprint(
            str(row["COMPOUND_SMILES"]),
            fp_type=fp_type,
            fp_size=fp_size,
        )
        solvent_array = row[SOLVENT_COLS].values.astype(float)
        x_i = np.concatenate([fp_array, solvent_array])
        x_list.append(x_i)
        y_list.append(float(row["Rf"]))

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float).reshape(-1, 1)

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def relu(z):
    """Apply ReLU activation element-wise."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Compute derivative of ReLU activation element-wise."""
    dz = np.zeros_like(z)
    dz[z > 0.0] = 1.0
    return dz


def sigmoid(z):
    """Apply sigmoid activation element-wise."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Compute derivative of sigmoid activation element-wise."""
    s = sigmoid(z)
    return s * (1.0 - s)


def linear(z):
    """Identity activation function."""
    return z


def linear_derivative(z):
    """Derivative of identity activation."""
    return np.ones_like(z)


class FullyConnectedLayer:
    """Simple fully connected layer with configurable activation."""

    def __init__(self, input_dim, output_dim, activation="relu"):
        """Initialize layer parameters and choose activation function."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_in = None
        self.z = None

        self.w = np.random.randn(input_dim, output_dim) * (1.0 / np.sqrt(input_dim))
        self.b = np.zeros((1, output_dim), dtype=float)

        if activation == "relu":
            self.activation = relu
            self.activation_deriv = relu_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid_derivative
        else:
            self.activation = linear
            self.activation_deriv = linear_derivative

    def forward(self, a_in):
        """Run forward pass for one dense layer."""
        self.a_in = a_in
        self.z = a_in @ self.w + self.b
        return self.activation(self.z)

    def backward(self, d_a_out):
        """Backpropagate gradients through this layer."""
        if self.a_in is None or self.z is None:
            raise RuntimeError("Backward called before forward.")

        d_z = d_a_out * self.activation_deriv(self.z)
        d_w = (self.a_in.T @ d_z) / d_z.shape[0]
        d_b = np.sum(d_z, axis=0, keepdims=True) / d_z.shape[0]
        d_a_in = d_z @ self.w.T
        return d_a_in, d_w, d_b


class NeuralNetwork:
    """Feed-forward neural network composed of fully connected layers."""

    def __init__(self, layers_config):
        """Build network layers from (in_dim, out_dim, activation) tuples."""
        self.layers_config = list(layers_config)
        self.layers = [
            FullyConnectedLayer(in_dim, out_dim, activation=act)
            for in_dim, out_dim, act in self.layers_config
        ]
        self.model_config = {}

    def forward(self, x):
        """Run forward pass through all layers."""
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, d_a):
        """Run backward pass through all layers and collect gradients."""
        grads = []
        for layer in reversed(self.layers):
            d_a, d_w, d_b = layer.backward(d_a)
            grads.append((d_w, d_b))
        grads.reverse()
        return grads

    def update_params(self, grads, lr):
        """Update all layer parameters using gradient descent."""
        for layer, (d_w, d_b) in zip(self.layers, grads):
            layer.w -= lr * d_w
            layer.b -= lr * d_b


def mse_loss(y_pred, y_true):
    """Compute mean squared error between predictions and targets."""
    return float(np.mean((y_pred - y_true) ** 2))


def clip_predictions(preds):
    """Clip predicted Rf values to the valid range [0, 1]."""
    return np.clip(preds, 0.0, 1.0)


def train_network(
    model,
    x_train,
    y_train,
    lr=0.001,
    max_epoch=100,
    batch_size=32,
    verbose=True,
):
    """Mini-batch training for regression."""
    n_samples = x_train.shape[0]
    loss_history = []

    for epoch in range(max_epoch):
        indices = np.random.permutation(n_samples)
        x_shuf = x_train[indices]
        y_shuf = y_train[indices]

        for i in range(0, n_samples, batch_size):
            x_batch = x_shuf[i : i + batch_size]
            y_batch = y_shuf[i : i + batch_size]

            y_pred = model.forward(x_batch)
            d_a_out = 2.0 * (y_pred - y_batch)
            grads = model.backward(d_a_out)
            model.update_params(grads, lr=lr)

        y_pred_full = model.forward(x_train)
        current_loss = mse_loss(y_pred_full, y_train)
        loss_history.append(current_loss)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch + 1:3d}/{max_epoch} | MSE Loss: {current_loss:.4f}")

    return loss_history


def build_layer_config(
    input_dim,
    hidden_dims,
    output_activation,
):
    """Create dense-layer configuration from hidden sizes and output activation."""
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append((prev_dim, h, "relu"))
        prev_dim = h
    layers.append((prev_dim, 1, output_activation))
    return layers


def save_model(
    model,
    model_path,
    model_config,
    train_info,
):
    """Always write a freshly trained model and overwrite any existing file."""
    path = _resolve_model_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        path.unlink()

    payload = {
        "layers_config": model.layers_config,
        "weights": [layer.w for layer in model.layers],
        "biases": [layer.b for layer in model.layers],
        "model_config": model_config,
        "train_info": train_info,
    }

    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_model(model_path):
    """Load a serialized TLC model and rebuild the neural network instance."""
    path = _resolve_model_path(model_path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    model = NeuralNetwork(payload["layers_config"])
    for layer, w, b in zip(model.layers, payload["weights"], payload["biases"]):
        layer.w = w
        layer.b = b

    model.model_config = payload.get("model_config", DEFAULT_CONFIG.copy())
    return model


def _train_model(
    dataset_path=None,
    model_config=None,
    verbose=True,
):
    """Train the TLC model and return the trained model with training summary."""
    config = DEFAULT_CONFIG.copy()
    if model_config:
        config.update(model_config)

    np.random.seed(int(config["random_state"]))

    df = load_clean_data(dataset_path)
    if verbose:
        print(f"Training rows used: {len(df)}")

    x_train, x_test, y_train, y_test = prepare_dataset(
        df,
        fp_type=str(config["fp_type"]),
        fp_size=int(config["fp_size"]),
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
    )

    layers_config = build_layer_config(
        input_dim=x_train.shape[1],
        hidden_dims=list(config["hidden_dims"]),
        output_activation=str(config["output_activation"]),
    )

    model = NeuralNetwork(layers_config)
    model.model_config = config

    train_network(
        model,
        x_train,
        y_train,
        lr=float(config["learning_rate"]),
        max_epoch=int(config["max_epoch"]),
        batch_size=int(config["batch_size"]),
        verbose=verbose,
    )

    y_pred_test = model.forward(x_test)
    if str(config["output_activation"]) == "linear":
        y_pred_test = clip_predictions(y_pred_test)

    test_mse = mse_loss(y_pred_test, y_test)
    train_info = {
        "dataset_rows": float(len(df)),
        "train_size": float(x_train.shape[0]),
        "test_size": float(x_test.shape[0]),
        "test_mse": float(test_mse),
    }

    if verbose:
        print(f"Finished training TLC model. Test MSE: {test_mse:.4f}")

    return model, train_info


def get_model(
    model_path="tlc_model.pkl",
    dataset_path=None,
    force_retrain=False,
    verbose=True,
):
    """Always retrain and overwrite model file at the target path."""
    resolved_model_path = _resolve_model_path(model_path)

    model, train_info = _train_model(
        dataset_path=dataset_path,
        model_config=DEFAULT_CONFIG,
        verbose=verbose,
    )
    save_model(model, resolved_model_path, model.model_config, train_info)

    if verbose:
        print(f"Saved trained TLC model to: {resolved_model_path}")

    return model


def predict_rf(
    model,
    smiles,
    hexane_pct,
    ea_pct,
):
    """Predict a molecule Rf value under binary Hexane/EA eluent."""
    hexane_pct = float(hexane_pct)
    ea_pct = float(ea_pct)

    if hexane_pct < 0 or ea_pct < 0:
        raise ValueError("Eluent fractions must be non-negative.")

    total = hexane_pct + ea_pct
    if total <= 0:
        raise ValueError("Hexane and EA cannot both be zero.")

    h_ratio = hexane_pct / total
    ea_ratio = ea_pct / total

    config = getattr(model, "model_config", DEFAULT_CONFIG)
    fp_type = str(config.get("fp_type", "rdkit"))
    fp_size = int(config.get("fp_size", 128))

    fp_array = smiles_to_fingerprint(smiles, fp_type=fp_type, fp_size=fp_size)
    solvent_array = np.array([h_ratio, ea_ratio, 0.0, 0.0, 0.0], dtype=float)

    x = np.concatenate([fp_array, solvent_array]).reshape(1, -1)
    pred = model.forward(x)
    pred = clip_predictions(pred)
    return float(pred[0, 0])


if __name__ == "__main__":
    _ = get_model(force_retrain=False, verbose=True)
