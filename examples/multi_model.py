# Import Libraries

import os.path

import matplotlib.pyplot as plt
import pandas as pd

import fm4m
import fm4m.datasets
from fm4m.constants import DATA_DIR, MODELS_PATH
from fm4m.logger import create_logger
from fm4m.path_utils import add_path

LOGGER = create_logger(__name__)

# %%

# 1 Load Data


train_df = pd.read_csv(os.path.join(DATA_DIR, "bace/train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "bace/test.csv"))

LOGGER.info(f"shape of train_df: {train_df.shape}")

INPUT = "smiles"
OUTPUT = "Class"

x_train = train_df[INPUT].to_list()
y_train = train_df[OUTPUT].to_list()

x_test = test_df[INPUT].to_list()
y_test = test_df[OUTPUT].to_list()

fm4m.datasets.avail_models()

# %%

# 2-2. Encoding

with add_path(MODELS_PATH):
    x_batch, x_batch_test = fm4m.get_vector_embeddings(
        x_train, x_test, model_type="MHG-GED", return_tensor=False
    )

LOGGER.info(f"x_batch shape: {x_batch.shape}, x_batch_test shape: {x_batch_test.shape}")

# 3. Model usage

fm4m.datasets.avail_downstream_models()

# %%

# 3-3. Example of multi-modal model usage

result = fm4m.multi_modal(
    model_list=["MHG-GED", "SMI-TED"],
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    downstream_model="DefaultClassifier",
)

LOGGER.info(f"result[0]: Result, '{result[0]}', {type(result[0])}")

# %%
# 4. Visualization

# 4-1. ROC-AUC: Classification task

fig, ax = plt.subplots()
ax.set_title("ROC-AUC Curve")
ax.plot(result[2], result[3], color="darkorange", lw=2, label=f"ROC curve (area = {result[1]:.4f})")
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend(loc="lower right")
