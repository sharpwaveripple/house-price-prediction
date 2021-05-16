import pandas as pd
from lightgbm import LGBMRegressor
import pickle
import os


def main():
    # load train and test data
    project_dir = "../../"
    processed_dir = os.path.join(project_dir, "data", "processed")
    train_fpath = os.path.join(processed_dir, "train.csv")
    test_fpath = os.path.join(processed_dir, "test.csv")

    train = pd.read_csv(train_fpath)
    test = pd.read_csv(test_fpath)

    X_ind = test[["Id"]]
    X_test = test[[x for x in test.columns if x != "Id"]]

    y_train = train["SalePrice"]
    X_train = train[[x for x in train.columns if x not in ["Id", "SalePrice"]]]

    random_state = 0

    estimator = LGBMRegressor(
        objective="regression_l2",
        boosting_type="goss",
        learning_rate=0.0148,
        n_estimators=805,
        random_state=random_state,
    )
    estimator.fit(X_train, y_train)

    scaler = pickle.load(open("../features/SalePrice_scaler.pkl", "rb"))
    y_pred = scaler.inverse_transform(estimator.predict(X_test))

    X_ind["SalePrice"] = y_pred

    outpath = os.path.join(processed_dir, "predictions.csv")
    print(f"Saving test predictions to {outpath}")
    X_ind.to_csv(outpath, index=None)


if __name__ == "__main__":
    main()
