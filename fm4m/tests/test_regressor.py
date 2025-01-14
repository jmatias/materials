import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier

from fm4m.config.datasets import clintox_dataset, esol_dataset
from fm4m.models.model import load_smi_ted_model


def test_solubility_regressor():
    train_df, test_df = esol_dataset()

    model = load_smi_ted_model()

    with torch.no_grad():
        X_train = model.encode(train_df["smiles"])
        X_test = model.encode(test_df["smiles"])

        y_train = train_df["prop"]
        y_test = test_df["prop"]

    assert X_train is not None
    assert X_test is not None

    regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)

    model = TransformedTargetRegressor(
        regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
    )

    model.fit(X_train, y_train)

    y_prob = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_prob)

    assert 0.0 < rmse < 1.0


def test_toxicity_classifier():
    train_df, test_df = clintox_dataset()
    model = load_smi_ted_model()

    with torch.no_grad():
        X_train = model.encode(train_df["smiles"])
        X_test = model.encode(test_df["smiles"])

        y_train = train_df["target"]
        y_test = test_df["target"]

    xgb_predict_concat = XGBClassifier()
    xgb_predict_concat.fit(X_train, y_train)
    y_prob = xgb_predict_concat.predict_proba(X_test)[
        :, 1
    ]  # get the probability of the positive class, drop the negative class
    roc_auc = roc_auc_score(y_test, y_prob)

    assert 0.0 < roc_auc < 1.0
