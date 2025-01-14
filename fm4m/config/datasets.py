from os.path import join

import pandas as pd
from fm4m.config.constants import DATA_DIR


def bace_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(join(DATA_DIR, "bace/train.csv"))
    test_df = pd.read_csv(join(DATA_DIR, "bace/test.csv"))
    return train_df, test_df


def clintox_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(join(DATA_DIR, "clintox/clintox_train.csv"))
    test_df = pd.read_csv(join(DATA_DIR, "clintox/clintox_test.csv"))
    return train_df, test_df


def esol_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(join(DATA_DIR, "esol/train.csv"))[["smiles", "prop"]]
    test_df = pd.read_csv(join(DATA_DIR, "esol/test.csv"))[["smiles", "prop"]]
    return train_df, test_df
