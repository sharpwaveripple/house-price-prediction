# -*- coding: utf-8 -*-
import os

import pandas as pd


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


def write_categorical(df, write=True):
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


project_dir = "../../"

df = read_data(project_dir)
write_categorical(df)
