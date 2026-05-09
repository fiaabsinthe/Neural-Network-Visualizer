import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



st.set_page_config(
    page_title="Neural Network Visualizer By Fia",
    layout="wide"
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #050505;
        color: #ffffff;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 48px;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 3px #ffffff, 0 0 3px #ffffff;
        margin-bottom: 0px;
    }
    
    .subtitle {
        font-size: 18px;
        color: #ffffff;
        opacity: 0.8;
        margin-bottom: 35px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .metric-card {
        background-color: #0d0d0d;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #333;
        /* The Neon Glow Border */
        box-shadow: inset 0 0 5px rgba(255, 0, 255, 0.2), 0 0 10px rgba(255, 0, 255, 0.1);
        transition: all 0.3s ease-in-out;
    }
    
    .metric-card:hover {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.4), inset 0 0 10px rgba(255, 0, 255, 0.2);
        transform: translateY(-2px);
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #ff00ff;
        margin-top: 40px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }

    .section-title::after {
        content: "";
        flex: 1;
        margin-left: 20px;
        height: 1px;
        background: linear-gradient(90deg, #ff00ff, transparent);
    }
    
    h3 {
        color: #00f2ff !important;
        margin-bottom: 5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_dataset(dataset_name: str, samples: int, noise: float, seed: int):
    if dataset_name == "Moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=samples, noise=noise, factor=0.5, random_state=seed)
    else:
        X, y = make_blobs(n_samples=samples, centers=2, cluster_std=1.5 + noise, random_state=seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_neurons, activation_name, output_size=2):
        super().__init__()
        self.activation_name = activation_name

        if activation_name == "ReLU":
            activation = nn.ReLU
        elif activation_name == "Tanh":
            activation = nn.Tanh
        else:
            activation = nn.Sigmoid

        layers = []
        current_size = input_size

        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_size, hidden_neurons))
            layers.append(activation())
            current_size = hidden_neurons

        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_layer_outputs(self, x):
        outputs = []
        current = x
        for layer in self.network:
            current = layer(current)
            if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                outputs.append(current.detach().cpu().numpy())
        return outputs


def train_model(model, X_train, y_train, X_test, y_test, learning_rate, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1).numpy()
                test_acc = accuracy_score(y_test, test_predictions)
                test_accuracies.append((epoch, test_acc))

    return train_losses, test_accuracies


def plot_dataset(X, y, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="inferno", edgecolors="k", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.25)
    return fig


def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=2)
    ax.set_title("Training Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    return fig


def plot_accuracy(test_accuracies):
    epochs = [item[0] for item in test_accuracies]
    accuracies = [item[1] for item in test_accuracies]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, accuracies, marker="o", linewidth=2)
    ax.set_title("Test Accuracy During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    return fig


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)

    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()

    zz = probabilities.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="coolwarm", alpha=0.65)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", alpha=0.9)
    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(contour, ax=ax, label="Class 1 Probability")
    return fig


def plot_activation_heatmap(layer_outputs, layer_index):
    if not layer_outputs:
        return None

    selected_output = layer_outputs[layer_index]
    preview = selected_output[:30]

    fig, ax = plt.subplots(figsize=(8, 4))
    heatmap = ax.imshow(preview, aspect="auto", cmap="inferno")
    ax.set_title(f"Hidden Layer {layer_index + 1} Activation Heatmap")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Sample")
    fig.colorbar(heatmap, ax=ax, label="Activation Value")
    return fig


def draw_architecture(input_size, hidden_layers, hidden_neurons, output_size):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")

    layer_sizes = [input_size] + [hidden_neurons] * hidden_layers + [output_size]
    layer_names = ["Input"] + [f"Hidden {i + 1}" for i in range(hidden_layers)] + ["Output"]

    x_positions = np.linspace(0.1, 0.9, len(layer_sizes))

    for layer_idx, (x, size) in enumerate(zip(x_positions, layer_sizes)):
        visible_neurons = min(size, 8)
        y_positions = np.linspace(0.15, 0.85, visible_neurons)

        for y in y_positions:
            circle = plt.Circle((x, y), 0.035, fill=True, alpha=0.85)
            ax.add_patch(circle)

        if size > 8:
            ax.text(x, 0.05, f"+ {size - 8} more", ha="center", fontsize=9)

        ax.text(x, 0.96, layer_names[layer_idx], ha="center", fontsize=11, fontweight="bold")
        ax.text(x, 0.90, f"{size} neurons", ha="center", fontsize=9)

    for i in range(len(layer_sizes) - 1):
        x1 = x_positions[i]
        x2 = x_positions[i + 1]
        y1_positions = np.linspace(0.15, 0.85, min(layer_sizes[i], 8))
        y2_positions = np.linspace(0.15, 0.85, min(layer_sizes[i + 1], 8))

        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([x1, x2], [y1, y2], alpha=0.08, linewidth=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig




st.markdown('<div class="main-title">Neural Network Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train a simple neural network and visualize how it learns patterns, forms decision boundaries, and transforms data through hidden layers!'
    'dont forget to explore !</div>',
    unsafe_allow_html=True
)


st.sidebar.header("Experiment Controls")

dataset_name = st.sidebar.selectbox("Dataset", ["Moons", "Circles", "Blobs"])
samples = st.sidebar.slider("Number of Samples", 100, 1000, 400, step=50)
noise = st.sidebar.slider("Dataset Noise", 0.01, 0.50, 0.20, step=0.01)

st.sidebar.divider()

hidden_layers = st.sidebar.slider("Hidden Layers", 1, 4, 2)
hidden_neurons = st.sidebar.slider("Neurons per Hidden Layer", 2, 64, 16, step=2)
activation_name = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])

st.sidebar.divider()

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
    value=0.01
)
epochs = st.sidebar.slider("Training Epochs", 100, 3000, 1000, step=100)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)

train_button = st.sidebar.button("Train Neural Network", use_container_width=True)



set_seed(seed)
X, y = generate_dataset(dataset_name, samples, noise, seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=seed, stratify=y
)



if "model" not in st.session_state:
    st.session_state.model = None
if "losses" not in st.session_state:
    st.session_state.losses = None
if "test_accuracies" not in st.session_state:
    st.session_state.test_accuracies = None
if "final_accuracy" not in st.session_state:
    st.session_state.final_accuracy = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None



if train_button:
    with st.spinner("Training neural network ..."):
        model = NeuralNetwork(
            input_size=2,
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            activation_name=activation_name,
            output_size=2
        )

        losses, test_accuracies = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            learning_rate,
            epochs
        )

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            test_outputs = model(X_test_tensor)
            predictions = torch.argmax(test_outputs, dim=1).numpy()
            final_accuracy = accuracy_score(y_test, predictions)

        st.session_state.model = model
        st.session_state.losses = losses
        st.session_state.test_accuracies = test_accuracies
        st.session_state.final_accuracy = final_accuracy
        st.session_state.predictions = predictions

    st.success("Model trained successfully.")



left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="section-title">1. Dataset</div>', unsafe_allow_html=True)
    st.pyplot(plot_dataset(X, y, f"{dataset_name} Dataset"), use_container_width=True)

with right_col:
    st.markdown('<div class="section-title">2. Neural Network Architecture</div>', unsafe_allow_html=True)
    st.pyplot(draw_architecture(2, hidden_layers, hidden_neurons, 2), use_container_width=True)

st.divider()



if st.session_state.model is not None:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    total_params = sum(p.numel() for p in st.session_state.model.parameters())
    final_loss = st.session_state.losses[-1]

    with metric_col1:
        st.metric("Final Accuracy", f"{st.session_state.final_accuracy * 100:.2f}%")
    with metric_col2:
        st.metric("Final Loss", f"{final_loss:.4f}")
    with metric_col3:
        st.metric("Train Samples", len(X_train))
    with metric_col4:
        st.metric("Model Parameters", total_params)

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown('<div class="section-title">3. Training Loss</div>', unsafe_allow_html=True)
        st.pyplot(plot_loss(st.session_state.losses), use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-title">4. Accuracy Progress</div>', unsafe_allow_html=True)
        st.pyplot(plot_accuracy(st.session_state.test_accuracies), use_container_width=True)

    st.divider()

    boundary_col, activation_col = st.columns([1, 1])

    with boundary_col:
        st.markdown('<div class="section-title">5. Decision Boundary</div>', unsafe_allow_html=True)
        st.pyplot(plot_decision_boundary(st.session_state.model, X, y), use_container_width=True)
        st.caption("The dark boundary line shows where the model changes its prediction between Class 0 and Class 1.")

    with activation_col:
        st.markdown('<div class="section-title">6. Hidden Layer Activations</div>', unsafe_allow_html=True)

        sample_tensor = torch.FloatTensor(X_test[:50])
        layer_outputs = st.session_state.model.get_layer_outputs(sample_tensor)

        if layer_outputs:
            selected_layer = st.selectbox(
                "Select hidden layer to inspect",
                list(range(1, len(layer_outputs) + 1)),
                format_func=lambda x: f"Hidden Layer {x}"
            )
            st.pyplot(plot_activation_heatmap(layer_outputs, selected_layer - 1), use_container_width=True)
            st.caption("Each row is a data sample. Each column is a neuron. Brighter/darker values show how strongly neurons activate.")
        else:
            st.info("No activation outputs available.")

    st.divider()

    report_col1, report_col2 = st.columns([1, 1])

    with report_col1:
        st.markdown('<div class="section-title">7. Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, st.session_state.predictions)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Class 0", "Actual Class 1"],
            columns=["Predicted Class 0", "Predicted Class 1"]
        )
        st.dataframe(cm_df, use_container_width=True)

    with report_col2:
        st.markdown('<div class="section-title">8. Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, st.session_state.predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">9. What This Shows</div>', unsafe_allow_html=True)
    st.write(
        """
        This visualizer demonstrates how a neural network learns to classify data by adjusting its internal weights during training.
        The dataset plot shows the raw input data, while the decision boundary shows how the trained model separates the two classes.
        The loss curve shows whether the model is improving over time, and the activation heatmap shows how hidden neurons respond to different inputs.
        """
    )

else:
    st.info("Configure the settings in the sidebar and click **Train Neural Network** to start the visualization")


st.divider()
st.caption("Built with Python, Streamlit, PyTorch, Scikit-learn, NumPy, Pandas, and Matplotlib.")
