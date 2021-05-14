# -*- coding: utf-8 -*-
import os

import pandas as pd

import get_important_features
from utils import encode_categoricals, split_data


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


def write_categoricals(df, write=True):
    categoricals = []
    for col in df.columns:
        if df[col].dtype == "object" or col == "MSSubClass":
            if col == "dataset":
                continue
            else:
                print(f"Guessing {col} is a categorical variable")
                categoricals.append(col)

    if write:
        fpath = os.path.join("./", "categoricals.txt")
        print(f"Writing categorical variables to {fpath}\n")
        open(fpath, "w+").writelines("\n".join(categoricals))


def filter_variables(df, project_dir):
    write_categoricals(df)
    fpath = os.path.join(project_dir, "src", "features", "X.txt")
    if os.path.exists(fpath):
        print(f"{fpath} found, loading")
    else:
        print(f"{fpath} not found, using feature importance pipeline")
        get_important_features.main()

    X_list = open(fpath, "r").read().splitlines()
    X_list.extend(["Id", "SalePrice", "dataset"])
    return X_list


def main():
    project_dir = "../../"
    df = read_data(project_dir)

    X_list = filter_variables(df, project_dir)
    X_list.remove("GarageYrBlt")

    # one hot encode and write out
    df_cat = encode_categoricals(df[X_list], project_dir, "one_hot")

    train, test = split_data(df_cat)
    for i, j in {'train': train, 'test': test}.items():
        fpath = os.path.join(project_dir, "data", "processed", f"{i}.csv")
        print(f"Saving processed {i}ing data to {fpath}")
        j.to_csv(fpath, index=None)
