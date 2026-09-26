"""
Customer Churn Prediction for a Telecom Company
CS 4407 - Machine Learning

Analyzes neural network structure and behavior via logic gates, then
builds a neural network classifier to predict telecom customer churn.
"""

import numpy as np
from sklearn.linear_model import Perceptron
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
    