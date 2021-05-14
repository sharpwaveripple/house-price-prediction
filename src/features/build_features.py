# -*- coding: utf-8 -*-
import os

import pandas as pd


def read_data(project_dir):
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"Reading imputed data from {fpath}")
    return pd.read_csv(fpath)


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
        fpath = os.path.join("./", "categoricals.txt")
        print(f"Writing categorical variables to {fpath}\n")
        open(fpath, "w+").writelines("\n".join(categoricals))

    return categoricals


project_dir = "../../"

df = read_data(project_dir)
check_categorical(df)
