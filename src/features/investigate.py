# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor, plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import seaborn as sns


def read_data(project_dir):
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"Reading imputed data from {fpath}")
    return pd.read_csv(fpath)


def calc_datesold(df, drop_orig_col=True):
    print("Creating DateSold from YrSold + MoSold * .01")
    df["DateSold"] = df["YrSold"] + df["MoSold"] * 0.01
    if drop_orig_col:
        print("Dropping YrSold and MoSold")
        df.drop(columns=["YrSold", "MoSold"], inplace=True)


# def drop_collinear(df):
#     drop_list = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
#     df.drop(columns=drop_list, inplace=True)


def check_categorical(df, write=True):
    categoricals = []
    for col in df.columns:
        if df[col].dtype == "object" or col == "MSSubClass":
            if col == "dataset":
                continue
            else:
                print(f"Guessing {col} is a categorical variable")
                categoricals.append(col)

    if write:
        fpath = "./categoricals.txt"
        print(f"Writing categorical variables to {fpath}\n")
        open(fpath, "w+").writelines("\n".join(categoricals))

    return categoricals


def encode_ordinal(data, categoricals):
    df = data.copy()
    enc = OrdinalEncoder()
    for col in categoricals:
        print(f"Encoding {col} using ordinal encoding")
        x = pd.Categorical(enc.fit_transform(df[[col]]).flatten())
        df[col] = x

    return df


def encode_one_hot(data, categoricals, drop_orig_col=True):
    df = data.copy()
    enc = OneHotEncoder(sparse=False)
    for col in categoricals:
        print(f"Encoding {col} using one hot encoding")
        enc_vals = enc.fit_transform(df[[col]])
        enc_keys = ["_".join([col, str(x)]) for x in enc.categories_[0]]
        enc_df = pd.DataFrame(enc_vals, columns=enc_keys, index=df.index)

        df = pd.concat([df, enc_df], axis=1)
        if drop_orig_col:
            df.drop(columns=[col], inplace=True)

    return df


def split_data(df):
    train = df[df["dataset"] == "train"].drop(columns="dataset")
    test = df[df["dataset"] == "test"].drop(columns="dataset")
    return train, test


project_dir = "../../"

df = read_data(project_dir)
cat = check_categorical(df)
calc_datesold(df)
# drop_collinear(df)

df_ord = encode_ordinal(df, cat)
# df_oh = encode_one_hot(df, cat)

train, test = split_data(df_ord)

y = train["SalePrice"]
X = train[[x for x in train.columns if x not in ["Id", "dataset", "SalePrice"]]]

sns.set_style("whitegrid")

for i in [0, 1, 42]:
    random_state = i

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    m = LGBMRegressor(objective="regression_l1")
    m.fit(X_train, y_train)

    names = pd.Series(m.feature_name_)
    importance = m.feature_importances_
    importance_thresh = importance > 50
    names[importance_thresh]

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_importance(m, max_num_features=20, ax=ax)
    fig.tight_layout()
    fig.savefig("./importance_plots/lgbm_train.png")

    sets = {"train": [X_train, y_train], "test": [X_test, y_test]}

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    n_features = 25
    for idx, split in enumerate(sets.keys()):
        ax = axes[idx]
        iv, dv = sets[split]

        perm = permutation_importance(
            m, iv, dv, n_repeats=50,
            scoring="neg_mean_absolute_error",
            random_state=random_state,
        )

        perm_df = pd.DataFrame(
            {"Feature": iv.columns, "Importance": perm.importances_mean}
        )
        perm_df.sort_values("Importance", ascending=False, inplace=True)

        sns.despine()
        sns.barplot(
            x="Importance",
            y="Feature",
            data=perm_df.iloc[0:n_features],
            color="C0",
            ax=ax,
        )
        ax.set_title(f"{split.capitalize()} set")
        # print(f" Saving output to ./importance_plots/perm_{split}.png")

    fig.tight_layout()
    fig.savefig(f"./importance_plots/perm_{i}.png")

    # if __name__ == "__main__":
    #     main()
