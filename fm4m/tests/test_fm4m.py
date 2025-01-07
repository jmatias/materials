from os.path import join

import pandas as pd
import pytest
import torch

from fm4m.main import get_representation, multi_modal, single_modal
from fm4m.constants import DATA_DIR, SMI_TED_MODEL


@pytest.mark.integration
@pytest.mark.slow
def test_multi_modal():
    # Arrange

    train_df = pd.read_csv(join(DATA_DIR, "bace/train.csv"))
    test_df = pd.read_csv(join(DATA_DIR, "bace/test.csv"))

    INPUT = "smiles"
    OUTPUT = "Class"

    # Act
    _t: tuple[torch.Tensor, torch.Tensor] = get_representation(
            train_df[INPUT], test_df[INPUT], model_type="MHG-GED", return_tensor=True
        )
    x_batch = _t[0]
    x_batch_test = _t[1]

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = multi_modal(
        model_list=["MHG-GED", "SMI-TED"],
        x_train=train_df[INPUT],
        y_train=train_df[OUTPUT],
        x_test=test_df[INPUT],
        y_test=test_df[OUTPUT],
        downstream_model="DefaultClassifier",
    )

    # Assert
    assert x_batch.shape[0] == train_df.shape[0]
    assert x_batch_test.shape[0] == test_df.shape[0]

    assert result is not None
    assert 0.0 < rmse_score < 1.0


def test_classifier_single_modal():
    # Arrange
    train_df = pd.read_csv(join(DATA_DIR, "clintox/clintox_train.csv"))
    test_df = pd.read_csv(join(DATA_DIR, "clintox/clintox_test.csv"))

    INPUT = "smiles"
    OUTPUT = "target"

    # Act

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = single_modal(
        SMI_TED_MODEL,
        x_train=train_df[INPUT],
        y_train=train_df[OUTPUT],
        x_test=test_df[INPUT],
        y_test=test_df[OUTPUT],
        downstream_model="DefaultClassifier",
    )

    # Assert

    assert result is not None
    assert 0.0 < rmse_score < 1.0
