# -*- coding: utf-8 -*-
import os

import pandas as pd
from sklearn.impute import SimpleImputer


def read_data(project_dir):
    # read raw train and test data and combine for ease of computation
    df_list = []
    for i in ["train", "test"]:
        fpath = os.path.join(project_dir, "data", "raw", f"{i}.csv")
        print(f"Reading raw {i}ing data from {fpath}")
        df = pd.read_csv(fpath)
        df["dataset"] = i
        df_list.append(df)

    print("Returning combined dataset")
    return pd.concat(df_list)


def impute_missing(df):
    """Impute missing values.
    There are 3 different strategies to handle different types of "missingness":
    0 (for when NaN's are numeric and mean 0),
    'None' (for when NaN's are categorical and mean 0), and
    most frequent (for when NaN's are categorical and sporadic).
    Lists corresponding to each of these are read in from ./imp/
    """

    # impute missing values. Note that there are three types of
    imputers = {
        "zero": SimpleImputer(strategy="constant", fill_value=0),
        "freq": SimpleImputer(strategy="most_frequent"),
        "none": SimpleImputer(strategy="constant", fill_value="None"),
    }

    for strategy, imputer in imputers.items():
        fpath = os.path.join("./", "imp", f"{strategy}.txt")
        print(f"\nReading imputation list from {fpath}")
        cols = open(fpath).read().splitlines()
        for col in cols:
            n_missing = df[col].isna().sum()
            print(f"Imputing {n_missing} values in {col} with {strategy}")
            df[col] = imputer.fit_transform(df[[col]])

    return df


def write_data(project_dir, df):
    # write interim imputed data
    fpath = os.path.join(project_dir, "data", "interim", "df.csv")
    print(f"\nSaving imputed data to {fpath}")
    df.to_csv(fpath, index=None)


def main():
    project_dir = "../../"
    df = read_data(project_dir)
    df = impute_missing(df)
    write_data(project_dir, df)


if __name__ == "__main__":
    main()
