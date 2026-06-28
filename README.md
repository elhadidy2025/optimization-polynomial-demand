# Optimization Methods for Polynomial Regression - Demand Forecasting

**Student:** Mohamed Elhadidy  

This project implements polynomial regression for retail demand forecasting and compares optimization methods under different objective structures.

## Dataset

The project uses the Kaggle Store Item Demand Forecasting data. The code expects:

```text
data/train.csv
```

with columns:

```text
date, store, item, sales
```

The submitted package already contains `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` folder.

## How to run

From the project root:

```bash
python src/main.py
```

The code uses :

```text
NumPy, pandas, matplotlib
```


## Output

Running the script creates/updates:

```text
figures/   convergence and accuracy plots
results/   CSV tables with experimental results
```

The final report is already included at:

```text
report/report.pdf
```

## Implemented methods

- Gradient Descent
- Nesterov acceleration
- Heavy Ball momentum
- Stochastic Gradient Descent
- SAGA
- Adagrad
- Adam
- Proximal Gradient / ISTA
- FISTA

## Objective functions

- Mean Squared Error: smooth convex
- MSE + L2 regularization: smooth strongly convex
- MSE + L1 regularization: convex non-smooth
- MSE + smooth L0-like regularization: smooth non-convex


