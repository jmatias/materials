from os.path import join

import pandas as pd
import pytest
import torch

from fm4m import get_vector_embeddings, multi_modal, single_modal
from fm4m.config import datasets
from fm4m.config.constants import DATA_DIR, SMI_TED_MODEL
from fm4m.models.model import DownstreamModelType


@pytest.fixture
def bace_dataset():
    return datasets.bace_dataset()


@pytest.fixture
def clintox_dataset():
    return datasets.clintox_dataset()


@pytest.mark.integration
@pytest.mark.slow
def test_multi_modal(bace_dataset):
    # Arrange
    train_df, test_df = bace_dataset

    # Act
    _t: tuple[torch.Tensor, torch.Tensor] = get_vector_embeddings(
        train_df["smiles"], test_df["smiles"], model_type="MHG-GED", return_tensor=True
    )
    x_batch = _t[0]
    x_batch_test = _t[1]

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = multi_modal(
        model_list=["MHG-GED", "SMI-TED"],
        x_train=train_df["smiles"],
        y_train=train_df["Class"],
        x_test=test_df["smiles"],
        y_test=test_df["Class"],
        downstream_model=DownstreamModelType.DefaultClassifier,
    )

    # Assert
    assert x_batch.shape[0] == train_df.shape[0]
    assert x_batch_test.shape[0] == test_df.shape[0]

    assert result is not None
    assert 0.0 < rmse_score < 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_classifier_single_modal(clintox_dataset):
    # Arrange

    train_df, test_df = clintox_dataset

    INPUT = "smiles"
    OUTPUT = "target"

    # Act

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = single_modal(
        SMI_TED_MODEL,
        x_train=train_df[INPUT],
        y_train=train_df[OUTPUT],
        x_test=test_df[INPUT],
        y_test=test_df[OUTPUT],
        downstream_model=DownstreamModelType.DefaultClassifier,
    )

    # Assert

    assert result is not None
    assert 0.0 < rmse_score < 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_regressor_single_modal(clintox_dataset):
    # Arrange

    train_df, test_df = clintox_dataset

    INPUT = "smiles"
    OUTPUT = "target"

    # Act

    result, rmse_score, y_batch_test, y_prob, class_0, class_1 = single_modal(
        SMI_TED_MODEL,
        x_train=train_df[INPUT],
        y_train=train_df[OUTPUT],
        x_test=test_df[INPUT],
        y_test=test_df[OUTPUT],
        downstream_model=DownstreamModelType.DefaultRegressor,
    )

    # Assert

    assert result is not None
    assert 0.0 < rmse_score < 1.0
