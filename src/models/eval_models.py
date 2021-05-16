'''
Simple first-pass script to evaluate which models look good at the outset.
Best performing model is further trained in train_model.py
'''

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def model_dx(m, X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state,
    )
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    rmse = np.round(mean_squared_error(y_test, y_pred, squared=False), 2)
    print(f"RMSE: {rmse}\n")


# load train data
df = pd.read_csv("../../data/processed/train.csv")

y = df["SalePrice"]
X = df[[x for x in df.columns if x not in ["Id", "SalePrice"]]]

random_state = 42
max_iter = 1e6
fit_intercept = False
loss = "squared_epsilon_insensitive"

print("Running LR")
LR = LinearRegression(fit_intercept=fit_intercept)
model_dx(LR, X, y, random_state)

print("Running SGD")
SGD = SGDRegressor(
    loss=loss,
    penalty="l2",
    max_iter=max_iter,
    fit_intercept=fit_intercept,
    verbose=0,
    random_state=random_state,
)
model_dx(SGD, X, y, random_state)

print("Running SVR")
SVR = LinearSVR(
    epsilon=0.001,
    C=100,
    loss=loss,
    fit_intercept=fit_intercept,
    max_iter=max_iter,
    verbose=0,
    random_state=random_state,
)
model_dx(SVR, X, y, random_state)

print("Running MLP")
MLP = MLPRegressor(
    hidden_layer_sizes=1000,
    solver="lbfgs",
    max_iter=max_iter,
    verbose=0,
    random_state=random_state,
)
model_dx(MLP, X, y, random_state)

print("Running LGBM")
LGBM = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    random_state=random_state,

)
model_dx(LGBM, X, y, random_state)
