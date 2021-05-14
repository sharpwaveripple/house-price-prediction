# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from utils import encode_categoricals

# set constants
RANDOM_STATES = [0, 1, 42, 981, 314159]
TEST_SIZE = 0.3
FEATURE_RECURRENCE = 3


def read_data(project_dir):
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"Reading imputed data from {fpath}")
    return pd.read_csv(fpath)


def split_data(df):
    train = df[df["dataset"] == "train"].drop(columns="dataset")
    test = df[df["dataset"] == "test"].drop(columns="dataset")
    return train, test


def fit_LGBM(X, y, objective="regression_l1", random_state=42):
    print(f"Fitting LGBM with {objective}")
    lgbm = LGBMRegressor(objective="regression_l1", random_state=random_state)
    lgbm.fit(X, y)
    return lgbm


def permute(model, X, y, thresh=100, n_repeats=20, random_state=42):
    print(f"Permuting data {n_repeats} times")
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
    )

    features, importances = X.columns, perm.importances_mean

    if thresh:
        print(f"Filtering features with importance < {thresh}")
        mask = importances >= thresh
        features, importances = features[mask], importances[mask]

    # format results as dataframe
    df = pd.DataFrame({"Feature": features, "Importance": importances})
    return df.sort_values("Importance", ascending=False)


def perm_barplot(data, fpath):
    "Takes output from permute(), plots it as a barplot, and saves it to fpath"
    fig, ax = plt.subplots(figsize=(22, 9))
    sns.barplot(x="Importance", y="Feature", data=data, color="C0", ax=ax)
    sns.despine(fig)
    fig.tight_layout()
    print(f"Saving permutation barplot to {fpath}")
    fig.savefig(fpath)


def iter_perm_importance(X, y, project_dir, save_png=True):
    """Calculate feature importance over multiple random states.

    To combat the stochastic nature of the algorithms used,
    calculate feature importance using multiple random_states.
    Only take features that survive across multiple random states.
    """
    counts = {x: 0 for x in X.columns}
    for random_state in RANDOM_STATES:
        print(f"\nRandom state = {random_state}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=random_state
        )

        # fit LGBM using l1 loss and calculate the permutations on the test set
        lgbm = fit_LGBM(X_train, y_train, random_state=random_state)
        df_perm = permute(lgbm, X_test, y_test)

        # iterate over surviving features and add them to count
        for row in df_perm.iterrows():
            feature = row[1]["Feature"]
            counts[feature] += 1

        # save barplot of results
        if save_png:
            outpath = os.path.join(project_dir, "reports", "figures")
            out_png = os.path.join(outpath, f"features_perm{random_state}.png")
            perm_barplot(df_perm, out_png)

    return counts


def select_important_features(counts, project_dir):
    # only take features that occur > FEATURE_RECURRENCE
    X_list = [x for x, y in counts.items() if y > FEATURE_RECURRENCE]
    print(f"\nFeatures that recurred > {FEATURE_RECURRENCE}:")
    [print(x) for x in X_list]
    fpath = os.path.join(project_dir, "src", "features", "X.txt")
    print(f"Writing important features to {fpath}\n")
    open(fpath, "w+").writelines("\n".join(X_list))


def main():
    project_dir = "../../"
    sns.set_style("whitegrid")

    df = read_data(project_dir)
    df_enc = encode_categoricals(df, project_dir)

    train, test = split_data(df_enc)
    y = train["SalePrice"]
    X = train[[x for x in train.columns if x not in ["Id", "dataset", "SalePrice"]]]

    counts = iter_perm_importance(X, y, project_dir)
    select_important_features(counts, project_dir)


if __name__ == "__main__":
    main()
