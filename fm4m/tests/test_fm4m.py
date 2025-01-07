import os

import pandas as pd
import pytest

import fm4m
from fm4m import get_representation
from fm4m.constants import MODELS_PATH, DATA_DIR
from fm4m.path_utils import add_path


@pytest.mark.integration
def test_multi_modal():

    train_df = pd.read_csv(os.path.join(DATA_DIR,"bace/train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR,"bace/test.csv"))

    INPUT = "smiles"
    OUTPUT = "Class"


    with add_path(MODELS_PATH):
        x_batch, x_batch_test = get_representation(
            train_df[INPUT], test_df[INPUT], model_type="MHG-GED", return_tensor=False
        )

    result = fm4m.multi_modal(
        model_list=["MHG-GED", "SMI-TED"],
        x_train=train_df[INPUT],
        y_train=train_df[OUTPUT],
        x_test=test_df[INPUT],
        y_test=test_df[OUTPUT],
        downstream_model="DefaultClassifier",


)

    print(f"result[0]: Result, '{result[0]}', {type(result[0])}")
