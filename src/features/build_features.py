# -*- coding: utf-8 -*-
import os

import pandas as pd

from . import get_important_features, utils
from pathlib import Path


def read_data(project_dir):
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"Reading imputed data from {fpath}")
    return pd.read_csv(fpath)


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


def main():
    project_dir = Path(__file__).resolve().parents[2]
    df = read_data(project_dir)

    X_list = filter_variables(df, project_dir)
    X_list.remove("GarageYrBlt")

    # one hot encode and write out
    df_cat = utils.encode_categoricals(df[X_list], project_dir, "one_hot")
    train, test = utils.scale_continuous(df_cat, project_dir)

    # train, test = split_data(df_cat)
    for i, j in {"train": train, "test": test}.items():
        fpath = os.path.join(project_dir, "data", "processed", f"{i}.csv")
        print(f"Saving processed {i}ing data to {fpath}")
        j.to_csv(fpath, index=None)


if __name__ == "__main__":
    main()
