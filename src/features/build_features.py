# -*- coding: utf-8 -*-
import os

import pandas as pd

import get_important_features
from utils import encode_categoricals, split_data
from sklearn.preprocessing import StandardScaler


def read_data(project_dir):
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"Reading imputed data from {fpath}")
    return pd.read_csv(fpath)


# def calc_datesold(df, drop_orig_col=True):
#     print("Creating DateSold from YrSold + MoSold * .01")
#     df["DateSold"] = df["YrSold"] + df["MoSold"] * 0.01
#     if drop_orig_col:
#         print("Dropping YrSold and MoSold")
#         df.drop(columns=["YrSold", "MoSold"], inplace=True)


# def drop_collinear(df):
#     drop_list = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
#     df.drop(columns=drop_list, inplace=True)


def write_vartypes(df, write=True):
    vartypes = {"continuous": [], "categorical": []}
    for col in df.columns:
        if col == "Id":
            continue
        elif df[col].dtype == "object" or col == "MSSubClass":
            if col == "dataset":
                continue
            else:
                print(f"Guessing {col} is a categorical variable")
                vartypes["categorical"].append(col)
        else:
            print(f"Guessing {col} is a continuous variable")
            vartypes["continuous"].append(col)

    if write:
        for vartype, varlist in vartypes.items():
            fpath = os.path.join("./", f"{vartype}.txt")
            print(f"Writing {vartype} variables to {fpath}\n")
            open(fpath, "w+").writelines("\n".join(varlist))


def filter_variables(df, project_dir):
    write_vartypes(df)
    fpath = os.path.join(project_dir, "src", "features", "X.txt")
    if os.path.exists(fpath):
        print(f"{fpath} found, loading")
    else:
        print(f"{fpath} not found, using feature importance pipeline")
        get_important_features.main()

    X_list = open(fpath, "r").read().splitlines()
    X_list.extend(["Id", "SalePrice", "dataset"])
    return X_list


def scale_continuous(df, project_dir):
    """Encode categorical variables in a dataframe.

    Two types of encoding are supported: ["ordinal", "one_hot"]

    This function assumes there is a list of categorical features in
    src/features/categoricals.txt, usually created by running
    src/features/build_features.py.
    """
    fpath = os.path.join(project_dir, "src", "features", "continuous.txt")
    continuous = open(fpath).read().splitlines()

    train, test = split_data(df)
    scaler = StandardScaler()
    for col in continuous:
        if col in train.columns:
            x_train = scaler.fit_transform(train[[col]])
            train[col] = x_train
            x_test = scaler.transform(test[[col]])
            test[col] = x_test

    return train, test


def main():
    project_dir = "../../"
    df = read_data(project_dir)

    X_list = filter_variables(df, project_dir)
    X_list.remove("GarageYrBlt")

    # one hot encode and write out
    df_cat = encode_categoricals(df[X_list], project_dir, "one_hot")
    train, test = scale_continuous(df_cat, project_dir)

    # train, test = split_data(df_cat)
    for i, j in {"train": train, "test": test}.items():
        fpath = os.path.join(project_dir, "data", "processed", f"{i}.csv")
        print(f"Saving processed {i}ing data to {fpath}")
        j.to_csv(fpath, index=None)


if __name__ == "__main__":
    main()
