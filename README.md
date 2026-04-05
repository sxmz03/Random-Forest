# Traffic Congestion Prediction — Random Forest

A Random Forest classifier that predicts traffic congestion levels based on vehicle count data, built using the Kaggle Traffic Prediction dataset.

## Overview

This project trains a Random Forest model to classify real-world traffic situations (e.g. low, normal, high, heavy) from time-series vehicle count data. It uses ordinal encoding for categorical features and evaluates model performance on a held-out test set.

## Features

- Random Forest multi-class classification
- Ordinal encoding of time, date, and day-of-week features
- 60/40 train/test split with reproducible random state
- Vehicle-type breakdown: cars, bikes, buses, trucks
- Seaborn and Matplotlib visualisations

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | Random Forest model and train/test split |
| category_encoders | Ordinal encoding of categorical features |
| Pandas / NumPy | Data loading and manipulation |
| Matplotlib / Seaborn | Data visualisation |

## Dataset

[Kaggle — Traffic Prediction Dataset](https://www.kaggle.com/datasets/hasanramezani/traffic-prediction-dataset)

Columns: `Time`, `Date`, `Day of the week`, `CarCount`, `BikeCount`, `BusCount`, `TruckCount`, `Total`, `Traffic Situation`

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders
```

Update the dataset path in `number-1.py` and run:

```bash
jupyter notebook number-1.py
# or open directly in Kaggle
```

## Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)
```
