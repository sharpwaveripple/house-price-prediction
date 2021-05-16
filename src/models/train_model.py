import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


# https://stackoverflow.com/questions/42228735/scikit-learn-gridsearchcv-with-multiple-repetitions/42230764#42230764
def nested_cv(
    estimator, search_spaces, X, y,
    scoring="neg_mean_squared_error",
    inner_cv=5, outer_cv=10, random_state=42,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state,
    )

    opt = BayesSearchCV(
        estimator=estimator, search_spaces=search_spaces,
        scoring="neg_mean_squared_error", n_iter=25, cv=inner_cv,
        verbose=0, n_jobs=4, random_state=random_state,
    )
    opt.fit(X_train, y_train)
    print("Best params:\n%s" % opt.best_params_)
    inner_cv_rmse = np.round(np.sqrt(opt.best_score_ * -1), 2)
    print(f"Inner loop RMSE: {inner_cv_rmse}")

    nested_score = cross_val_score(
        opt, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=4,
    )
    outer_cv_rmse = np.round(np.sqrt(nested_score.mean() * -1), 2)
    print(f"Outer loop RMSE: {outer_cv_rmse}")

    y_pred = opt.predict(X_test)
    rmse = np.round(mean_squared_error(y_test, y_pred, squared=False), 2)
    print(f"Validation RMSE: {rmse}")
    return opt


def main():
    # load train and test data
    project_dir = Path(__file__).resolve().parents[2]
    fpath = os.path.join(project_dir, "data", "processed", "train.csv")

    df = pd.read_csv(fpath)

    y = df["SalePrice"]
    X = df[[x for x in df.columns if x not in ["Id", "SalePrice"]]]

    random_state = 0

    print("Tuning LGBMRegressor hyperparameters using nested cv")
    estimator = LGBMRegressor(
        boosting_type="gbdt", random_state=random_state
    )
    search_spaces = {
        "n_estimators": Integer(100, 1000),
        "learning_rate": Real(1e-4, 1e-1),
    }
    opt = nested_cv(estimator, search_spaces, X, y, random_state=random_state)

    model_out = os.path.join(project_dir, "models", "LGBM.pkl")
    print(f"Saving LGBM model to {model_out}")
    pickle.dump(opt, open(model_out, "wb"))


if __name__ == "__main__":
    main()
