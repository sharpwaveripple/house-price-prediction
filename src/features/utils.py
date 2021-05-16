# -*- coding: utf-8 -*-
import os
import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def split_data(df):
    train = df[df["dataset"] == "train"].drop(columns="dataset")
    test = df[df["dataset"] == "test"].drop(columns=["dataset", "SalePrice"])
    return train, test


def encode_ordinal(data, categoricals):
    df = data.copy()
    enc = OrdinalEncoder()
    for col in categoricals:
        if col in df.columns:
            print(f"Encoding {col} using ordinal encoding")
            x = pd.Categorical(enc.fit_transform(df[[col]]).flatten())
            df[col] = x

    return df


def encode_one_hot(data, categoricals, drop_orig_col=True):
    df = data.copy()
    enc = OneHotEncoder(sparse=False)
    for col in categoricals:
        if col in df.columns:
            print(f"Encoding {col} using one hot encoding")
            enc_vals = enc.fit_transform(df[[col]])
            enc_keys = ["_".join([col, str(x)]) for x in enc.categories_[0]]
            enc_df = pd.DataFrame(enc_vals, columns=enc_keys, index=df.index)

            df = pd.concat([df, enc_df], axis=1)
            if drop_orig_col:
                df.drop(columns=[col], inplace=True)

    return df


def encode_categoricals(df, project_dir, encoding="ordinal"):
    """Encode categorical variables in a dataframe.

    Two types of encoding are supported: ["ordinal", "one_hot"]

    This function assumes there is a list of categorical features in
    src/features/categoricals.txt, usually created by running
    src/features/build_features.py.
    """
    fpath = os.path.join(project_dir, "src", "features", "categorical.txt")
    categoricals = open(fpath).read().splitlines()
    if encoding == "ordinal":
        df_enc = encode_ordinal(df, categoricals)
    elif encoding == "one_hot":
        df_enc = encode_one_hot(df, categoricals)

    return df_enc


def scale_continuous(df, project_dir):
    """Scale continuous variables to mean of 0 and unit variance.

    This function assumes there is a list of continuous features in
    src/features/continuous.txt, usually created by running
    src/features/build_features.py.
    """
    fpath = os.path.join(project_dir, "src", "features", "continuous.txt")
    continuous = open(fpath).read().splitlines()

    train, test = split_data(df)
    scaler = StandardScaler()
    for col in continuous:
        # don't fit transforms to test set to avoid data leakage
        if col in train.columns:
            x_train = scaler.fit_transform(train[[col]])
            train[col] = x_train
            if col != "SalePrice":
                x_test = scaler.transform(test[[col]])
                test[col] = x_test
            else:
                scaler_out = f"./{col}_scaler.pkl"
                print(f"Saving scaler for {col} to {scaler_out}")
                pickle.dump(scaler, open(scaler_out, "wb"))

    return train, test
