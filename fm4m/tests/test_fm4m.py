from os.path import join
from pprint import pprint

import pandas as pd
import pytest
import torch

import fm4m
from fm4m import get_representation
from fm4m.constants import MODELS_PATH, DATA_DIR
from fm4m.path_utils import add_path


@pytest.mark.integration
@pytest.mark.slow
def test_multi_modal():
    # Arrange

    train_df = pd.read_csv(join(DATA_DIR, "bace/train.csv"))
    test_df = pd.read_csv(join(DATA_DIR, "bace/test.csv"))

    INPUT = "smiles"
    OUTPUT = "Class"

    # Act
    with add_path(MODELS_PATH):

        _t: tuple[torch.Tensor, torch.Tensor] = get_representation(
            train_df[INPUT], test_df[INPUT], model_type="MHG-GED", return_tensor=True)
        x_batch = _t[0]
        x_batch_test = _t[1]

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = fm4m.multi_modal(
        model_list=["MHG-GED", "SMI-TED"], x_train=train_df[INPUT], y_train=train_df[OUTPUT],
        x_test=test_df[INPUT], y_test=test_df[OUTPUT], downstream_model="DefaultClassifier", )

    # Assert
    assert x_batch.shape[0] == train_df.shape[0]
    assert x_batch_test.shape[0] == test_df.shape[0]

    assert result is not None
    assert 0.0 < rmse_score < 1.0
