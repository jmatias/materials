import json

import pandas as pd


def avail_models_data():
    global datasets
    global models

    datasets = [
        {
            "Dataset": "hiv",
            "Input": "smiles",
            "Output": "HIV_active",
            "Path": "data/hiv",
            "Timestamp": "2024-06-26 11:27:37",
        },
        {
            "Dataset": "esol",
            "Input": "smiles",
            "Output": "ESOL predicted log solubility in mols per litre",
            "Path": "data/esol",
            "Timestamp": "2024-06-26 11:31:46",
        },
        {
            "Dataset": "freesolv",
            "Input": "smiles",
            "Output": "expt",
            "Path": "data/freesolv",
            "Timestamp": "2024-06-26 11:33:47",
        },
        {
            "Dataset": "lipo",
            "Input": "smiles",
            "Output": "y",
            "Path": "data/lipo",
            "Timestamp": "2024-06-26 11:34:37",
        },
        {
            "Dataset": "bace",
            "Input": "smiles",
            "Output": "Class",
            "Path": "data/bace",
            "Timestamp": "2024-06-26 11:36:40",
        },
        {
            "Dataset": "bbbp",
            "Input": "smiles",
            "Output": "p_np",
            "Path": "data/bbbp",
            "Timestamp": "2024-06-26 11:39:23",
        },
        {
            "Dataset": "clintox",
            "Input": "smiles",
            "Output": "CT_TOX",
            "Path": "data/clintox",
            "Timestamp": "2024-06-26 11:42:43",
        },
    ]

    models = [
        {
            "Name": "bart",
            "Model Name": "SELFIES-TED",
            "Description": "BART model for string based SELFIES modality",
            "Timestamp": "2024-06-21 12:32:20",
        },
        {
            "Name": "mol-xl",
            "Model Name": "MolFormer",
            "Description": "MolFormer model for string based SMILES modality",
            "Timestamp": "2024-06-21 12:35:56",
        },
        {
            "Name": "mhg",
            "Model Name": "MHG-GED",
            "Description": "Molecular hypergraph model",
            "Timestamp": "2024-07-10 00:09:42",
        },
        {
            "Name": "smi-ted",
            "Model Name": "SMI-TED",
            "Description": "SMILES based encoder decoder model",
            "Timestamp": "2024-07-10 00:09:42",
        },
    ]


def avail_models(raw=False):
    global models

    models = [
        {
            "Name": "smi-ted",
            "Model Name": "SMI-TED",
            "Description": "SMILES based encoder decoder model",
        },
        {
            "Name": "bart",
            "Model Name": "SELFIES-TED",
            "Description": "BART model for string based SELFIES modality",
        },
        {
            "Name": "mol-xl",
            "Model Name": "MolFormer",
            "Description": "MolFormer model for string based SMILES modality",
        },
        {"Name": "mhg", "Model Name": "MHG-GED", "Description": "Molecular hypergraph model"},
        {
            "Name": "Mordred",
            "Model Name": "Mordred",
            "Description": "Baseline: A descriptor-calculation software application that can calculate more than 1800 two- and three-dimensional descriptors",
        },
        {
            "Name": "MorganFingerprint",
            "Model Name": "MorganFingerprint",
            "Description": "Baseline: Circular atom environments based descriptor",
        },
    ]

    if raw:
        return models
    else:
        return pd.DataFrame(models).drop("Name", axis=1)

    return models


def avail_downstream_models(raw=False):
    global downstream_models

    downstream_models = [
        {"Name": "XGBClassifier", "Task Type": "Classfication"},
        {"Name": "DefaultClassifier", "Task Type": "Classfication"},
        {"Name": "SVR", "Task Type": "Regression"},
        {"Name": "Kernel Ridge", "Task Type": "Regression"},
        {"Name": "Linear Regression", "Task Type": "Regression"},
        {"Name": "DefaultRegressor", "Task Type": "Regression"},
    ]

    if raw:
        return downstream_models
    else:
        return pd.DataFrame(downstream_models)


def avail_datasets():
    global datasets

    datasets = [
        {
            "Dataset": "hiv",
            "Input": "smiles",
            "Output": "HIV_active",
            "Path": "data/hiv",
            "Timestamp": "2024-06-26 11:27:37",
        },
        {
            "Dataset": "esol",
            "Input": "smiles",
            "Output": "ESOL predicted log solubility in mols per litre",
            "Path": "data/esol",
            "Timestamp": "2024-06-26 11:31:46",
        },
        {
            "Dataset": "freesolv",
            "Input": "smiles",
            "Output": "expt",
            "Path": "data/freesolv",
            "Timestamp": "2024-06-26 11:33:47",
        },
        {
            "Dataset": "lipo",
            "Input": "smiles",
            "Output": "y",
            "Path": "data/lipo",
            "Timestamp": "2024-06-26 11:34:37",
        },
        {
            "Dataset": "bace",
            "Input": "smiles",
            "Output": "Class",
            "Path": "data/bace",
            "Timestamp": "2024-06-26 11:36:40",
        },
        {
            "Dataset": "bbbp",
            "Input": "smiles",
            "Output": "p_np",
            "Path": "data/bbbp",
            "Timestamp": "2024-06-26 11:39:23",
        },
        {
            "Dataset": "clintox",
            "Input": "smiles",
            "Output": "CT_TOX",
            "Path": "data/clintox",
            "Timestamp": "2024-06-26 11:42:43",
        },
    ]

    return datasets


def reset():
    """datasets = {"esol": ["smiles", "ESOL predicted log solubility in mols per litre", "data/esol", "2024-06-26 11:36:46.509324"],
    "freesolv": ["smiles", "expt", "data/freesolv", "2024-06-26 11:37:37.393273"],
    "lipo": ["smiles", "y", "data/lipo", "2024-06-26 11:37:37.393273"],
    "hiv": ["smiles", "HIV_active", "data/hiv",  "2024-06-26 11:37:37.393273"],
    "bace": ["smiles", "Class", "data/bace", "2024-06-26 11:38:40.058354"],
    "bbbp": ["smiles", "p_np", "data/bbbp","2024-06-26 11:38:40.058354"],
    "clintox": ["smiles", "CT_TOX", "data/clintox","2024-06-26 11:38:40.058354"],
    "sider": ["smiles","1:", "data/sider","2024-06-26 11:38:40.058354"],
    "tox21": ["smiles",":-2", "data/tox21","2024-06-26 11:38:40.058354"]
    }"""

    datasets = [
        {
            "Dataset": "hiv",
            "Input": "smiles",
            "Output": "HIV_active",
            "Path": "data/hiv",
            "Timestamp": "2024-06-26 11:27:37",
        },
        {
            "Dataset": "esol",
            "Input": "smiles",
            "Output": "ESOL predicted log solubility in mols per litre",
            "Path": "data/esol",
            "Timestamp": "2024-06-26 11:31:46",
        },
        {
            "Dataset": "freesolv",
            "Input": "smiles",
            "Output": "expt",
            "Path": "data/freesolv",
            "Timestamp": "2024-06-26 11:33:47",
        },
        {
            "Dataset": "lipo",
            "Input": "smiles",
            "Output": "y",
            "Path": "data/lipo",
            "Timestamp": "2024-06-26 11:34:37",
        },
        {
            "Dataset": "bace",
            "Input": "smiles",
            "Output": "Class",
            "Path": "data/bace",
            "Timestamp": "2024-06-26 11:36:40",
        },
        {
            "Dataset": "bbbp",
            "Input": "smiles",
            "Output": "p_np",
            "Path": "data/bbbp",
            "Timestamp": "2024-06-26 11:39:23",
        },
        {
            "Dataset": "clintox",
            "Input": "smiles",
            "Output": "CT_TOX",
            "Path": "data/clintox",
            "Timestamp": "2024-06-26 11:42:43",
        },
        # {"Dataset": "sider", "Input": "smiles", "Output": "1:", "path": "data/sider", "Timestamp": "2024-06-26 11:38:40.058354"},
        # {"Dataset": "tox21", "Input": "smiles", "Output": ":-2", "path": "data/tox21", "Timestamp": "2024-06-26 11:38:40.058354"}
    ]

    models = [
        {
            "Name": "bart",
            "Description": "BART model for string based SELFIES modality",
            "Timestamp": "2024-06-21 12:32:20",
        },
        {
            "Name": "mol-xl",
            "Description": "MolFormer model for string based SMILES modality",
            "Timestamp": "2024-06-21 12:35:56",
        },
        {"Name": "mhg", "Description": "MHG", "Timestamp": "2024-07-10 00:09:42"},
        {
            "Name": "spec-gru",
            "Description": "Spectrum modality with GRU",
            "Timestamp": "2024-07-10 00:09:42",
        },
        {
            "Name": "spec-lstm",
            "Description": "Spectrum modality with LSTM",
            "Timestamp": "2024-07-10 00:09:54",
        },
        {
            "Name": "3d-vae",
            "Description": "VAE model for 3D atom positions",
            "Timestamp": "2024-07-10 00:10:08",
        },
    ]

    downstream_models = [
        {
            "Name": "XGBClassifier",
            "Description": "XG Boost Classifier",
            "Timestamp": "2024-06-21 12:31:20",
        },
        {
            "Name": "XGBRegressor",
            "Description": "XG Boost Regressor",
            "Timestamp": "2024-06-21 12:32:56",
        },
        {
            "Name": "2-FNN",
            "Description": "A two layer feedforward network",
            "Timestamp": "2024-06-24 14:34:16",
        },
        {
            "Name": "3-FNN",
            "Description": "A three layer feedforward network",
            "Timestamp": "2024-06-24 14:38:37",
        },
    ]

    with open("datasets.json", "w") as outfile:
        json.dump(datasets, outfile)

    with open("models.json", "w") as outfile:
        json.dump(models, outfile)

    with open("downstream_models.json", "w") as outfile:
        json.dump(downstream_models, outfile)


def update_downstream_model_list(list_model):
    # models[list_model[0]] = list_model[1]

    with open("downstream_models.json", "w") as outfile:
        json.dump(list_model, outfile)

    avail_models_data()


def update_model_list(list_model):
    # models[list_model[0]] = list_model[1]

    with open("models.json", "w") as outfile:
        json.dump(list_model, outfile)

    avail_models_data()


def update_data_list(list_data):
    # datasets[list_data[0]] = list_data[1:]

    with open("datasets.json", "w") as outfile:
        json.dump(datasets, outfile)

    avail_models_data()
