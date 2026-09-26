"""
Customer Churn Prediction for a Telecom Company
CS 4407 - Machine Learning

Analyzes neural network structure and behavior via logic gates, then
builds a neural network classifier to predict telecom customer churn.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Step 1 / Question 1.iii: AND, OR, NOT gates with a single neuron
# ---------------------------------------------------------------------------
# A single neuron (Perceptron) can learn AND/OR/NOT since all three are
# linearly separable - one straight line can divide their true/false cases.

print("=" * 70)
print("QUESTION 1.iii: AND, OR, NOT GATES (SINGLE NEURON)")
print("=" * 70)

and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs = np.array([0, 0, 0, 1])

or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs = np.array([0, 1, 1, 1])

not_inputs = np.array([[0], [1]])
not_outputs = np.array([1, 0])

for gate_name, inputs, outputs in [
    ("AND", and_inputs, and_outputs),
    ("OR", or_inputs, or_outputs),
    ("NOT", not_inputs, not_outputs),
]:
    model = Perceptron(random_state=42)  # fixed seed for reproducible results
    model.fit(inputs, outputs)
    predictions = model.predict(inputs)

    print(f"\n{gate_name} gate:")
    table_rows = [
        list(row) + [expected, predicted]
        for row, expected, predicted in zip(inputs, outputs, predictions)
    ]
    headers = [f"Input {i + 1}" for i in range(inputs.shape[1])] + ["Expected", "Predicted"]
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

# ---------------------------------------------------------------------------
# Step 2 / Question 1.iii: XOR with a hidden-layer neural network
# ---------------------------------------------------------------------------
# XOR isn't linearly separable, so a single neuron can't learn it (shown
# below for comparison); a hidden layer lets the network combine multiple
# linear boundaries into a non-linear one.

print("\n" + "=" * 70)
print("QUESTION 1.iii: XOR GATE (HIDDEN-LAYER NETWORK)")
print("=" * 70)

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([0, 1, 1, 0])

xor_hidden_model = MLPClassifier(
    hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", max_iter=5000, random_state=42
)
xor_hidden_model.fit(xor_inputs, xor_outputs)
xor_hidden_predictions = xor_hidden_model.predict(xor_inputs)

print("\nXOR gate (hidden-layer network):")
table_rows = [
    list(row) + [expected, predicted]
    for row, expected, predicted in zip(xor_inputs, xor_outputs, xor_hidden_predictions)
]
print(tabulate(table_rows, headers=["Input 1", "Input 2", "Expected", "Predicted"], tablefmt="fancy_grid"))

# Single neuron attempting XOR, for comparison - expected to fail, since
# XOR is not linearly separable.
xor_single_neuron_model = Perceptron(random_state=42)
xor_single_neuron_model.fit(xor_inputs, xor_outputs)
xor_single_neuron_predictions = xor_single_neuron_model.predict(xor_inputs)

print("\nXOR gate (single neuron, for comparison):")
table_rows = [
    list(row) + [expected, predicted]
    for row, expected, predicted in zip(xor_inputs, xor_outputs, xor_single_neuron_predictions)
]
print(tabulate(table_rows, headers=["Input 1", "Input 2", "Expected", "Predicted"], tablefmt="fancy_grid"))

# Decision boundary comparison across all gates: AND and OR (both
# linearly separable, single neuron succeeds) next to XOR failing with a
# single neuron and succeeding with a hidden layer. One consistent color
# scheme throughout: a correctly classified point blends into its
# region's color, a misclassified point visibly clashes against it.
and_model = Perceptron(random_state=42)  # fixed seed for reproducible results
and_model.fit(and_inputs, and_outputs)

or_model = Perceptron(random_state=42)  # fixed seed for reproducible results
or_model.fit(or_inputs, or_outputs)

xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

panels = [
    (axes[0, 0], and_model, and_inputs, and_outputs, "AND (single neuron)"),
    (axes[0, 1], or_model, or_inputs, or_outputs, "OR (single neuron)"),
    (axes[1, 0], xor_single_neuron_model, xor_inputs, xor_outputs, "XOR (single neuron - fails)"),
    (axes[1, 1], xor_hidden_model, xor_inputs, xor_outputs, "XOR (hidden layer - solves)"),
]

for ax, model, gate_inputs, gate_outputs, title in panels:
    predictions_grid = model.predict(grid_points).reshape(xx.shape)
    ax.contourf(xx, yy, predictions_grid, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.6, vmin=0, vmax=1)
    ax.scatter(
        gate_inputs[:, 0], gate_inputs[:, 1], c=gate_outputs, cmap="coolwarm",
        edgecolors="black", s=120, zorder=3, vmin=0, vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")

plt.tight_layout()
plt.savefig("logic_gates_decision_boundaries.png")
print("\nSaved plot: logic_gates_decision_boundaries.png")

# ---------------------------------------------------------------------------
# Step 3 / Question 2.i: Load and inspect the churn dataset
# ---------------------------------------------------------------------------
# Loaded from a local file (downloaded from Kaggle) rather than a URL, so
# the project stays reproducible without any external network dependency.

print("\n" + "=" * 70)
print("QUESTION 2.i: LOAD AND INSPECT CHURN DATASET")
print("=" * 70)

churn_df = pd.read_csv("./data/Telco-Customer-Churn.csv")

print("\nDataset shape:", churn_df.shape)
print("\nColumn dtypes:")
print(churn_df.dtypes)
print("\nChurn distribution:")
print(churn_df["Churn"].value_counts())

# ---------------------------------------------------------------------------
# Step 4 / Question 2.i: Handle categorical variables
# ---------------------------------------------------------------------------
# customerID is just an identifier, not a predictive feature.
churn_df = churn_df.drop(columns=["customerID"])

# TotalCharges loaded as text because of blank values for customers with
# tenure=0 (new customers who haven't been billed yet) - convert to
# numeric and fill those with 0.
churn_df["TotalCharges"] = pd.to_numeric(churn_df["TotalCharges"], errors="coerce")
print(f"\nRows with missing TotalCharges before fill: {churn_df['TotalCharges'].isna().sum()}")
churn_df["TotalCharges"] = churn_df["TotalCharges"].fillna(0)

# Binary Yes/No columns (and gender) mapped to 1/0. SeniorCitizen is
# already 0/1 numeric, so it needs no mapping.
binary_columns = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
binary_mapping = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
for column_name in binary_columns:
    churn_df[column_name] = churn_df[column_name].map(binary_mapping)

# Remaining multi-category columns one-hot encoded.
multi_category_columns = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
churn_df = pd.get_dummies(churn_df, columns=multi_category_columns)

print("\nDataset shape after encoding:", churn_df.shape)
print("\nColumn dtypes after encoding:")
print(churn_df.dtypes.value_counts())

# ---------------------------------------------------------------------------
# Step 5 / Question 2.i: Scale features
# ---------------------------------------------------------------------------
# Only the continuous numeric columns need scaling - binary and one-hot
# encoded columns are already 0/1.

print("\n" + "=" * 70)
print("QUESTION 2.i: SCALE FEATURES")
print("=" * 70)

numeric_columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

print("\nFeature values before scaling:")
print(churn_df[numeric_columns_to_scale].describe())

scaler = StandardScaler()
churn_df[numeric_columns_to_scale] = scaler.fit_transform(churn_df[numeric_columns_to_scale])

print("\nFeature values after scaling:")
print(churn_df[numeric_columns_to_scale].describe())

# ---------------------------------------------------------------------------
# Step 6 / Question 2.i: Train/test split
# ---------------------------------------------------------------------------
# Stratified by Churn to preserve the real ~73.5%/26.5% class balance in
# both splits.

print("\n" + "=" * 70)
print("QUESTION 2.i: TRAIN/TEST SPLIT")
print("=" * 70)

X = churn_df.drop(columns=["Churn"])
y = churn_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")
print("\nTraining set churn distribution:")
print(y_train.value_counts(normalize=True).round(4))
print("\nTest set churn distribution:")
print(y_test.value_counts(normalize=True).round(4))

# ---------------------------------------------------------------------------
# Step 7 / Question 2.ii.a: Define the neural network architecture
# ---------------------------------------------------------------------------
# Two hidden layers (16, then 8 neurons) - enough capacity for the ~40
# input features without being excessive for a binary classification
# task. ReLU and adam are standard defaults at this dataset size (unlike
# the earlier 4-sample XOR problem, where adam got stuck and lbfgs/tanh
# was needed instead).

print("\n" + "=" * 70)
print("QUESTION 2.ii.a: DEFINE NEURAL NETWORK ARCHITECTURE")
print("=" * 70)

churn_model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,  # fixed seed for reproducible results
)

print("\nNetwork architecture:")
print(f"  Input features: {X_train.shape[1]}")
print(f"  Hidden layers: {churn_model.hidden_layer_sizes}")
print(f"  Activation function: {churn_model.activation}")
print("  Output: 1 neuron (binary classification - churn or not)")

# ---------------------------------------------------------------------------
# Step 8 / Question 2.ii.b: Train the model
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION 2.ii.b: TRAIN THE MODEL")
print("=" * 70)

churn_model.fit(X_train, y_train)

print(f"\nTraining completed after {churn_model.n_iter_} iterations")
print(f"Final training loss: {churn_model.loss_:.4f}")
