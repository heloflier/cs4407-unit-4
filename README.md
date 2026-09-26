# Customer Churn Prediction for a Telecom Company

Programming assignment: analyzing basic artificial neural network
structure and behavior (logic gates), then building a neural network
classifier to predict telecom customer churn, using `scikit-learn`.

## Contents

- `churn_neural_network.py` — main analysis script (logic gates,
  preprocessing, model, evaluation)
- `data/Telco-Customer-Churn.csv` — dataset (Telco Customer Churn),
  downloaded from HuggingFace and committed here for full
  reproducibility without any external download step
- `requirements.txt` — Python dependencies

## Workflow

The script is built incrementally, corresponding to the assignment's
questions:

1. Logic gates (Question 1.iii): AND/OR/NOT with a single neuron, XOR
   with a hidden-layer network
2. Data preprocessing for the churn dataset (Question 2.i): categorical
   encoding, feature scaling, train/test split
3. Neural network model (Question 2.ii): architecture, training,
   predictions
4. Model evaluation (Question 3): accuracy, confusion matrix,
   performance results

Question 1.i and 1.ii (neural network structure and forward/backward
propagation) are written analysis only, covered in the accompanying
Word document rather than in this script.

## Running

```bash
pip install -r requirements.txt
python churn_neural_network.py
```