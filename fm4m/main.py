import pickle

import mordred.error
import numpy as np
import pandas as pd
import torch
import umap
import xgboost as xgb
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

import fm4m.models.mhg_model.load as mhg
from .config.constants import (
    BART_MODEL,
    MHG_MODEL,
    MODELS_PATH,
    MODEL_ALIASES,
    MOL_XL_MODEL,
    MORDRED_MODEL,
    MORGAN_FINGERPRINT,
    SMI_TED_MODEL,
)
from .config.model_files import MHG_MODEL_PICKLE, MOL_FORMER_XL_BOTH_10PCT_PRETRAINED_MODEL
from .config.repository import avail_datasets, avail_models, avail_models_data
from .logger import create_logger
from .models.model import DownstreamModelType, ModelType
from .models.selfies_ted import Selfies as Bart
from .models.smi_ted.smi_ted_light.load import load_smi_ted
from .path_utils import add_path

datasets = {}
models = {}
downstream_models = {}

LOGGER = create_logger(__name__)

avail_models_data()


def get_vector_embeddings(
    train_data: pd.Series | list,
    test_data: pd.Series | list,
    model_type: ModelType | str,
    return_tensor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the embedding representation of the data using the specified model
    Args:
        train_data:
        test_data:
        model_type:
        return_tensor:

    Returns:

    """
    with add_path(MODELS_PATH):
        if model_type in MODEL_ALIASES:
            model_type = MODEL_ALIASES[model_type]

        if model_type == MHG_MODEL:
            x_batch, x_batch_test = load_mhg_model(test_data, train_data, return_tensor)

        elif model_type == BART_MODEL:
            x_batch, x_batch_test = load_bart_model(test_data, train_data, return_tensor)

        elif model_type == SMI_TED_MODEL:
            x_batch, x_batch_test = load_smi_ted_model(test_data, train_data, return_tensor)

        elif model_type == MOL_XL_MODEL:
            x_batch, x_batch_test = load_mol_xl_model(test_data, train_data, return_tensor)

        elif model_type == MORDRED_MODEL:
            x_batch, x_batch_test = load_mordered_model(test_data, train_data)

        elif model_type == MORGAN_FINGERPRINT:
            x_batch, x_batch_test = load_morgan_fingerprint_model(test_data, train_data)

    return x_batch, x_batch_test


def load_morgan_fingerprint_model(test_data, train_data):
    params = {"radius": 2, "nBits": 1024}
    mol_train = [Chem.MolFromSmiles(sm) for sm in train_data]
    mol_test = [Chem.MolFromSmiles(sm) for sm in test_data]
    x_batch = []
    for mol in mol_train:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, **params, bitInfo=info)
        vector = list(fp)
        x_batch.append(vector)
    x_batch = pd.DataFrame(x_batch)
    x_batch_test = []
    for mol in mol_test:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, **params, bitInfo=info)
        vector = list(fp)
        x_batch_test.append(vector)
    x_batch_test = pd.DataFrame(x_batch_test)
    return x_batch, x_batch_test


def load_mordered_model(test_data, train_data):
    all_data = train_data + test_data
    calc = Calculator(descriptors, ignore_3D=True)
    mol_list = [Chem.MolFromSmiles(sm) for sm in all_data]
    x_all = calc.pandas(mol_list)
    print(f"original mordred fv dim: {x_all.shape}")
    for j in x_all.columns:
        for k in range(len(x_all[j])):
            i = x_all.loc[k, j]
            if type(i) is mordred.error.Missing or type(i) is mordred.error.Error:
                x_all.loc[k, j] = np.nan
    x_all.dropna(how="any", axis=1, inplace=True)
    print(f"Nan excluded mordred fv dim: {x_all.shape}")
    x_batch = x_all.iloc[: len(train_data)]
    x_batch_test = x_all.iloc[
        len(train_data) :
    ]  # print(f'x_batch: {len(x_batch)}, x_batch_test: {len(x_batch_test)}')
    return x_batch, x_batch_test


def load_mol_xl_model(test_data, train_data, return_tensor):
    model = AutoModel.from_pretrained(
        MOL_FORMER_XL_BOTH_10PCT_PRETRAINED_MODEL,
        deterministic_eval=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MOL_FORMER_XL_BOTH_10PCT_PRETRAINED_MODEL, trust_remote_code=True
    )
    train_data = train_data.to_list() if isinstance(train_data, pd.Series) else train_data
    test_data = test_data.to_list() if isinstance(test_data, pd.Series) else test_data
    inputs = tokenizer(train_data.to_list(), padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    x_batch = outputs.pooler_output
    inputs = tokenizer(test_data.to_list(), padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    x_batch_test = outputs.pooler_output
    if not return_tensor:
        x_batch = pd.DataFrame(x_batch)
        x_batch_test = pd.DataFrame(x_batch_test)
    return x_batch, x_batch_test


def load_smi_ted_model(test_data, train_data, return_tensor):
    model = load_smi_ted()
    with torch.no_grad():
        x_batch = model.encode(train_data, return_torch=return_tensor)
        x_batch_test = model.encode(test_data, return_torch=return_tensor)
    return x_batch, x_batch_test


def load_bart_model(test_data, train_data, return_tensor):
    model = Bart()
    model.load()
    x_batch = model.encode(train_data, return_tensor=return_tensor)
    x_batch_test = model.encode(test_data, return_tensor=return_tensor)
    return x_batch, x_batch_test


def load_mhg_model(test_data, train_data, return_tensor):
    model = mhg.load(MHG_MODEL_PICKLE)
    with torch.no_grad():
        train_emb = model.encode(train_data)
        x_batch = torch.stack(train_emb)

        test_emb = model.encode(test_data)
        x_batch_test = torch.stack(test_emb)
    if not return_tensor:
        x_batch = pd.DataFrame(x_batch)
        x_batch_test = pd.DataFrame(x_batch_test)
    return x_batch, x_batch_test


# noinspection t
def single_modal(
    model: str,
    dataset=None,
    downstream_model: str | DownstreamModelType = None,
    params=None,
    x_train=None,
    x_test=None,
    y_train=None,
    y_test=None,
):
    LOGGER.debug(model)

    data = avail_models(raw=True)
    df = pd.DataFrame(data)
    # print(list(df["Name"].values))

    if model in df["Name"].to_list():
        model_type = model
    elif MODEL_ALIASES[model] in df["Name"].to_list():
        model_type = MODEL_ALIASES[model]
    else:
        LOGGER.warning("Model not available")
        return None

    data = avail_datasets()
    df = pd.DataFrame(data)

    task = "Dummy"
    if dataset in df["Dataset"].to_list():
        task = dataset
        with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
            x_batch, y_batch, x_batch_test, y_batch_test = pickle.load(f1)
        LOGGER.debug(f" Representation loaded successfully")

    elif x_train is None:
        x_batch, x_batch_test, y_batch, y_batch_test = _load_custom_dataset(dataset, model_type)

        LOGGER.debug(f" Representation loaded successfully")

    else:
        y_batch = y_train
        y_batch_test = y_test
        x_batch, x_batch_test = get_vector_embeddings(x_train, x_test, model_type)

    # exclude row containing Nan value
    if isinstance(x_batch, torch.Tensor):
        x_batch = pd.DataFrame(x_batch)
    nan_indices = x_batch.index[x_batch.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch.dropna(inplace=True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch[index]
        LOGGER.debug(f"x_batch Nan index: {nan_indices}")
        LOGGER.debug(f"x_batch shape: {x_batch.shape}, y_batch len: {len(y_batch)}")

    if isinstance(x_batch_test, torch.Tensor):
        x_batch_test = pd.DataFrame(x_batch_test)
    nan_indices = x_batch_test.index[x_batch_test.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch_test.dropna(inplace=True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch_test[index]
        LOGGER.debug(f"x_batch_test Nan index: {nan_indices}")
        LOGGER.debug(
            f"x_batch_test shape: {x_batch_test.shape}, y_batch_test len: {len(y_batch_test)}"
        )

    LOGGER.info(f" Calculating ROC AUC Score ...")

    if downstream_model == DownstreamModelType.XGBClassifier:
        return build_xgb_classifier_model(
            model_type, params, task, x_batch, x_batch_test, y_batch, y_batch_test
        )

    elif downstream_model == DownstreamModelType.DefaultClassifier:
        return build_default_classifier_model(
            model_type, task, x_batch, x_batch_test, y_batch, y_batch_test
        )

    elif downstream_model == DownstreamModelType.SVR:
        return build_svr_model(params, x_batch, x_batch_test, y_batch, y_batch_test)

    elif downstream_model == DownstreamModelType.KernelRidge:
        return build_kernel_ridge_model(params, x_batch, x_batch_test, y_batch, y_batch_test)

    elif downstream_model == DownstreamModelType.LinearRegression:
        return build_linear_regression_model(params, x_batch, x_batch_test, y_batch, y_batch_test)

    elif downstream_model == DownstreamModelType.DefaultRegressor:
        return build_default_regressor_model(x_batch, x_batch_test, y_batch, y_batch_test)


def _load_custom_dataset(dataset, model_type):
    LOGGER.info("Custom Dataset")
    # return
    components = dataset.split(",")
    train_data = pd.read_csv(components[0])[components[2]]
    test_data = pd.read_csv(components[1])[components[2]]
    y_batch = pd.read_csv(components[0])[components[3]]
    y_batch_test = pd.read_csv(components[1])[components[3]]
    x_batch, x_batch_test = get_vector_embeddings(train_data, test_data, model_type)
    return x_batch, x_batch_test, y_batch, y_batch_test


def build_xgb_classifier_model(
    model_type, params, task, x_batch, x_batch_test, y_batch, y_batch_test
):
    if params is None:
        xgb_predict_concat = XGBClassifier()
    else:
        xgb_predict_concat = XGBClassifier(
            **params
        )  # n_estimators=5000, learning_rate=0.01, max_depth=10
    xgb_predict_concat.fit(x_batch, y_batch)
    y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]
    roc_auc = roc_auc_score(y_batch_test, y_prob)
    fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
    LOGGER.info(f"ROC-AUC Score: {roc_auc:.4f}")
    try:
        with open(f"plot_emb/{task}_{model_type}.pkl", "rb") as f1:
            class_0, class_1 = pickle.load(f1)
    except:
        LOGGER.debug("Generating latent plots")
        reducer = umap.UMAP(
            metric="euclidean",
            n_neighbors=10,
            n_components=2,
            low_memory=True,
            min_dist=0.1,
            verbose=False,
        )
        n_samples = np.minimum(1000, len(x_batch))

        try:
            x = y_batch.values[:n_samples]
        except:
            x = y_batch[:n_samples]
        index_0 = [index for index in range(len(x)) if x[index] == 0]
        index_1 = [index for index in range(len(x)) if x[index] == 1]

        try:
            features_umap = reducer.fit_transform(x_batch[:n_samples])
            class_0 = features_umap[index_0]
            class_1 = features_umap[index_1]
        except:
            class_0 = []
            class_1 = []
        LOGGER.debug("Generating latent plots : Done")
    # vizualize(roc_auc,fpr, tpr, x_batch, y_batch )
    result = f"ROC-AUC Score: {roc_auc:.4f}"
    return result, roc_auc, fpr, tpr, class_0, class_1


def build_default_classifier_model(model_type, task, x_batch, x_batch_test, y_batch, y_batch_test):
    xgb_predict_concat = XGBClassifier()  # n_estimators=5000, learning_rate=0.01, max_depth=10
    xgb_predict_concat.fit(x_batch, y_batch)
    y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]
    roc_auc = roc_auc_score(y_batch_test, y_prob)
    fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
    LOGGER.info(f"ROC-AUC Score: {roc_auc:.4f}")
    try:
        with open(f"plot_emb/{task}_{model_type}.pkl", "rb") as f1:
            class_0, class_1 = pickle.load(f1)
    except:
        LOGGER.info("Generating latent plots")
        reducer = umap.UMAP(
            metric="euclidean",
            n_neighbors=10,
            n_components=2,
            low_memory=True,
            min_dist=0.1,
            verbose=False,
        )
        n_samples = np.minimum(1000, len(x_batch))

        try:
            x = y_batch.values[:n_samples]
        except:
            x = y_batch[:n_samples]

        try:
            features_umap = reducer.fit_transform(x_batch[:n_samples])
            index_0 = [index for index in range(len(x)) if x[index] == 0]
            index_1 = [index for index in range(len(x)) if x[index] == 1]

            class_0 = features_umap[index_0]
            class_1 = features_umap[index_1]
        except:
            class_0 = []
            class_1 = []

        LOGGER.debug("Generating latent plots : Done")
    # vizualize(roc_auc,fpr, tpr, x_batch, y_batch )
    result = f"ROC-AUC Score: {roc_auc:.4f}"
    return result, roc_auc, fpr, tpr, class_0, class_1


def build_kernel_ridge_model(params, x_batch, x_batch_test, y_batch, y_batch_test):
    if params == None:
        regressor = KernelRidge()
    else:
        regressor = KernelRidge(**params)
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
    ).fit(x_batch, y_batch)
    y_prob = model.predict(x_batch_test)
    RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
    LOGGER.info(f"RMSE Score: {RMSE_score:.4f}")
    result = f"RMSE Score: {RMSE_score:.4f}"
    LOGGER.info("Generating latent plots")
    reducer = umap.UMAP(
        metric="euclidean",
        n_neighbors=10,
        n_components=2,
        low_memory=True,
        min_dist=0.1,
        verbose=False,
    )
    n_samples = np.minimum(1000, len(x_batch))
    features_umap = reducer.fit_transform(x_batch[:n_samples])
    try:
        x = y_batch.values[:n_samples]
    except:
        x = y_batch[:n_samples]
    # index_0 = [index for index in range(len(x)) if x[index] == 0]
    # index_1 = [index for index in range(len(x)) if x[index] == 1]
    class_0 = features_umap  # [index_0]
    class_1 = features_umap  # [index_1]
    LOGGER.debug("Generating latent plots : Done")
    return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


def build_svr_model(params, x_batch, x_batch_test, y_batch, y_batch_test):
    if params == None:
        regressor = SVR()
    else:
        regressor = SVR(**params)
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
    ).fit(x_batch, y_batch)
    y_prob = model.predict(x_batch_test)
    RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
    LOGGER.info(f"RMSE Score: {RMSE_score:.4f}")
    result = f"RMSE Score: {RMSE_score:.4f}"
    LOGGER.info("Generating latent plots")
    reducer = umap.UMAP(
        metric="euclidean",
        n_neighbors=10,
        n_components=2,
        low_memory=True,
        min_dist=0.1,
        verbose=False,
    )
    n_samples = np.minimum(1000, len(x_batch))
    try:
        x = y_batch.values[:n_samples]
    except:
        x = y_batch[:n_samples]
    # index_0 = [index for index in range(len(x)) if x[index] == 0]
    # index_1 = [index for index in range(len(x)) if x[index] == 1]
    try:
        features_umap = reducer.fit_transform(x_batch[:n_samples])
        class_0 = features_umap  # [index_0]
        class_1 = features_umap  # [index_1]
    except:
        class_0 = []
        class_1 = []
    LOGGER.debug("Generating latent plots : Done")
    return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


def build_linear_regression_model(params, x_batch, x_batch_test, y_batch, y_batch_test):
    if params == None:
        regressor = LinearRegression()
    else:
        regressor = LinearRegression(**params)
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
    ).fit(x_batch, y_batch)
    y_prob = model.predict(x_batch_test)
    RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
    LOGGER.info(f"RMSE Score: {RMSE_score:.4f}")
    result = f"RMSE Score: {RMSE_score:.4f}"
    LOGGER.info("Generating latent plots")
    reducer = umap.UMAP(
        metric="euclidean",
        n_neighbors=10,
        n_components=2,
        low_memory=True,
        min_dist=0.1,
        verbose=False,
    )
    n_samples = np.minimum(1000, len(x_batch))
    features_umap = reducer.fit_transform(x_batch[:n_samples])
    try:
        x = y_batch.values[:n_samples]
    except:
        x = y_batch[:n_samples]
    # index_0 = [index for index in range(len(x)) if x[index] == 0]
    # index_1 = [index for index in range(len(x)) if x[index] == 1]
    class_0 = features_umap  # [index_0]
    class_1 = features_umap  # [index_1]
    LOGGER.debug("Generating latent plots : Done")
    return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


def build_default_regressor_model(x_batch, x_batch_test, y_batch, y_batch_test):
    regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
    ).fit(x_batch, y_batch)
    y_prob = model.predict(x_batch_test)
    RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
    LOGGER.info(f"RMSE Score: {RMSE_score:.4f}")
    result = f"RMSE Score: {RMSE_score:.4f}"
    LOGGER.info("Generating latent plots")
    reducer = umap.UMAP(
        metric="euclidean",
        n_neighbors=10,
        n_components=2,
        low_memory=True,
        min_dist=0.1,
        verbose=False,
    )
    n_samples = np.minimum(1000, len(x_batch))
    features_umap = reducer.fit_transform(x_batch[:n_samples])
    try:
        x = y_batch.values[:n_samples]
    except:
        x = y_batch[:n_samples]
    # index_0 = [index for index in range(len(x)) if x[index] == 0]
    # index_1 = [index for index in range(len(x)) if x[index] == 1]
    class_0 = features_umap  # [index_0]
    class_1 = features_umap  # [index_1]
    LOGGER.debug("Generating latent plots : Done")
    return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


# noinspection t
def multi_modal(
    model_list,
    dataset=None,
    downstream_model: DownstreamModelType | str = None,
    params=None,
    x_train=None,
    x_test=None,
    y_train=None,
    y_test=None,
):
    # print(model_list)
    data = avail_datasets()
    df = pd.DataFrame(data)

    if dataset in list(df["Dataset"].values):
        task = dataset
        predefined = True
    elif x_train is None:
        predefined = False
        components = dataset.split(",")
        train_data = pd.read_csv(components[0])[components[2]]
        test_data = pd.read_csv(components[1])[components[2]]

        y_batch = pd.read_csv(components[0])[components[3]]
        y_batch_test = pd.read_csv(components[1])[components[3]]

        LOGGER.debug("Custom Dataset loaded")
    else:
        predefined = False
        y_batch = y_train
        y_batch_test = y_test
        train_data = x_train
        test_data = x_test

    alias = {
        "MHG-GED": "mhg",
        "SELFIES-TED": "bart",
        "MolFormer": "mol-xl",
        "Molformer": "mol-xl",
        "SMI-TED": "smi-ted",
        "Mordred": "Mordred",
        "MorganFingerprint": "MorganFingerprint",
    }
    # if set(model_list).issubset(list(df["Name"].values)):
    if set(model_list).issubset(list(alias.keys())):
        for i, model in enumerate(model_list):
            if model in alias.keys():
                model_type = alias[model]
            else:
                model_type = model

            if i == 0:
                if predefined:
                    with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
                        x_batch, y_batch, x_batch_test, y_batch_test = pickle.load(f1)
                    LOGGER.debug(f" Loaded representation/{task}_{model_type}.pkl")
                else:
                    x_batch, x_batch_test = get_vector_embeddings(train_data, test_data, model_type)
                    x_batch = pd.DataFrame(x_batch)
                    x_batch_test = pd.DataFrame(x_batch_test)

            else:
                if predefined:
                    with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
                        x_batch_1, y_batch_1, x_batch_test_1, y_batch_test_1 = pickle.load(f1)
                        LOGGER.debug(f" Loaded representation/{task}_{model_type}.pkl")
                else:
                    x_batch_1, x_batch_test_1 = get_vector_embeddings(
                        train_data, test_data, model_type
                    )
                    x_batch_1 = pd.DataFrame(x_batch_1)
                    x_batch_test_1 = pd.DataFrame(x_batch_test_1)

                x_batch = pd.concat([x_batch, x_batch_1], axis=1)
                x_batch_test = pd.concat([x_batch_test, x_batch_test_1], axis=1)

    else:
        LOGGER.error("Model not available")
        return

    num_columns = x_batch_test.shape[1]
    x_batch_test.columns = [f"{i + 1}" for i in range(num_columns)]

    num_columns = x_batch.shape[1]
    x_batch.columns = [f"{i + 1}" for i in range(num_columns)]

    # exclude row containing Nan value
    if isinstance(x_batch, torch.Tensor):
        x_batch = pd.DataFrame(x_batch)
    nan_indices = x_batch.index[x_batch.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch.dropna(inplace=True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch[index]
        LOGGER.debug(f"x_batch Nan index: {nan_indices}")
        LOGGER.debug(f"x_batch shape: {x_batch.shape}, y_batch len: {len(y_batch)}")

    if isinstance(x_batch_test, torch.Tensor):
        x_batch_test = pd.DataFrame(x_batch_test)
    nan_indices = x_batch_test.index[x_batch_test.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch_test.dropna(inplace=True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch_test[index]
        LOGGER.debug(f"x_batch_test Nan index: {nan_indices}")
        LOGGER.debug(
            f"x_batch_test shape: {x_batch_test.shape}, y_batch_test len: {len(y_batch_test)}"
        )

    LOGGER.debug(f"Representations loaded successfully")
    try:
        with open(f"plot_emb/{task}_multi.pkl", "rb") as f1:
            class_0, class_1 = pickle.load(f1)
    except:
        LOGGER.info("Generating latent plots")
        reducer = umap.UMAP(
            metric="euclidean",
            n_neighbors=10,
            n_components=2,
            low_memory=True,
            min_dist=0.1,
            verbose=False,
        )
        n_samples = np.minimum(1000, len(x_batch))
        features_umap = reducer.fit_transform(x_batch[:n_samples])

        if "Classifier" in str(downstream_model):
            try:
                x = y_batch.values[:n_samples]
            except:
                x = y_batch[:n_samples]
            index_0 = [index for index in range(len(x)) if x[index] == 0]
            index_1 = [index for index in range(len(x)) if x[index] == 1]

            class_0 = features_umap[index_0]
            class_1 = features_umap[index_1]

        else:
            class_0 = features_umap
            class_1 = features_umap

        LOGGER.debug("Generating latent plots : Done")

    LOGGER.info(f" Calculating ROC AUC Score ...")

    if downstream_model == DownstreamModelType.XGBClassifier:
        if params is None:
            xgb_predict_concat = XGBClassifier()
        else:
            xgb_predict_concat = XGBClassifier(
                **params
            )  # n_estimators=5000, learning_rate=0.01, max_depth=10)
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]

        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        LOGGER.info(f"ROC-AUC Score: {roc_auc:.4f}")

        # vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        # vizualize(x_batch_test, y_batch_test)
        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc, fpr, tpr, class_0, class_1

    elif downstream_model == DownstreamModelType.DefaultClassifier:
        xgb_predict_concat = XGBClassifier()  # n_estimators=5000, learning_rate=0.01, max_depth=10)
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]

        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        LOGGER.info(f"ROC-AUC Score: {roc_auc:.4f}")

        # vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        # vizualize(x_batch_test, y_batch_test)
        LOGGER.info(f"ROC-AUC Score: {roc_auc:.4f}")
        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc, fpr, tpr, class_0, class_1

    elif downstream_model == DownstreamModelType.SVR:
        if params == None:
            regressor = SVR()
        else:
            regressor = SVR(**params)
        model = TransformedTargetRegressor(
            regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
        ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        LOGGER.info("{RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1

    elif downstream_model == DownstreamModelType.LinearRegression:
        if params == None:
            regressor = LinearRegression()
        else:
            regressor = LinearRegression(**params)
        model = TransformedTargetRegressor(
            regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
        ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        LOGGER.info("{RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1

    elif downstream_model == DownstreamModelType.KernelRidge:
        if params is None:
            regressor = KernelRidge()
        else:
            regressor = KernelRidge(**params)
        model = TransformedTargetRegressor(
            regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
        ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        LOGGER.info("{RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1

    elif downstream_model == DownstreamModelType.DefaultRegressor:
        regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
        model = TransformedTargetRegressor(
            regressor=regressor, transformer=MinMaxScaler(feature_range=(-1, 1))
        ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        LOGGER.info(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


def finetune_optuna(x_batch, y_batch, x_batch_test, y_test):
    LOGGER.info(f" Finetuning with Optuna and calculating ROC AUC Score ...")
    X_train = x_batch.values
    y_train = y_batch.values
    X_test = x_batch_test.values
    y_test = y_test.values

    def objective(trial):
        # Define parameters to be optimized
        params = {  # 'objective': 'binary:logistic',
            "eval_metric": "auc",
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            # 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            # 'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 12),
            # 'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
            # 'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            # 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        }

        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain)

        # Predict probabilities
        y_pred = model.predict(dtest)

        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        LOGGER.info("ROC_AUC : ", roc_auc)

        return roc_auc
