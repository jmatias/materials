import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fm4m
from fm4m.constants import DATA_DIR
from fm4m.logger import create_logger

# Import Libraries

LOGGER = create_logger(__name__)

#%%

# 1 Load Data

train_df = pd.read_csv(os.path.join(DATA_DIR,"bace/train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR,"bace/test.csv"))

LOGGER.info(f"shape of train_df: {train_df.shape}")
train_df.head(3)

input = "smiles"
output = "Class"

xtrain = train_df[input].to_list()
ytrain = train_df[output].to_list()

xtest = test_df[input].to_list()
ytest = test_df[output].to_list()


fm4m.avail_models()

#%%

# 2-2. Encoding

x_batch, x_batch_test = fm4m.get_representation(
    xtrain, xtest, model_type="MHG-GED", return_tensor=False
)

LOGGER.info(f"x_batch shape: {x_batch.shape}, x_batch_test shape: {x_batch_test.shape}")

# 3. Model usage

fm4m.avail_downstream_models()

#%%

# 3-2. Example of single-modal model usage

result = fm4m.single_modal(
    model="MHG-GED",
    x_train=xtrain,
    y_train=ytrain,
    x_test=xtest,
    y_test=ytest,
    downstream_model="DefaultClassifier",
)

LOGGER.info(f"result[0]: Result, '{result[0]}', {type(result[0])}")
LOGGER.info(f"result[1]: Row score, {result[1]}, {type(result[1])}")
LOGGER.info(f"result[2]: False Positive Rate, type {type(result[2])}")
LOGGER.info(f"result[3]: True Positive Rate, type {type(result[3])}")
LOGGER.info(f"result[4]: Class_0 latent space, type {type(result[4])}")
LOGGER.info(f"result[5]: Class_1 latent space, type {type(result[5])}")

# 3-3. Example of multi-modal model usage

result = fm4m.multi_modal(
    model_list=["MHG-GED", "SMI-TED"],
    x_train=xtrain,
    y_train=ytrain,
    x_test=xtest,
    y_test=ytest,
    downstream_model="DefaultClassifier",
)

LOGGER.info(f"result[0]: Result, '{result[0]}', {type(result[0])}")

#%%
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


# 4-2. Latent space: Classification task

fig, ax = plt.subplots()
ax.set_title("T-SNE Plot")

class_0 = result[4]
class_1 = result[5]

plt.scatter(class_1[:, 0], class_1[:, 1], c="red", label="Class 1")
plt.scatter(class_0[:, 0], class_0[:, 1], c="blue", label="Class 0")

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(loc="lower right")

ax.set_title("Dataset Distribution")

#%%
# 4-3. Parity Plot: Regression task

train_df = pd.read_csv(os.path.join(DATA_DIR,"esol/train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR,"esol/test.csv"))

train_df.head(2)

input = "smiles"
output = "prop"

xtrain = train_df[input].to_list()
ytrain = train_df[output].to_list()

xtest = test_df[input].to_list()
ytest = test_df[output].to_list()

# can set hyperparameter for downstream-model
params = {"kernel": "rbf", "C": 2.0}
result = fm4m.single_modal(
    model="MHG-GED",
    x_train=xtrain,
    y_train=ytrain,
    x_test=xtest,
    y_test=ytest,
    downstream_model="SVR",
    params=params,
)

LOGGER.info(f"result[0]: Result, '{result[0]}', {type(result[0])}")
LOGGER.info(f"result[1]: Row score, {result[1]}, {type(result[1])}")
LOGGER.info(f"result[2]: Actual property values, type {type(result[2])}")
LOGGER.info(f"result[3]: Predicted property values, type {type(result[3])}")
LOGGER.info(f"result[4] & result[5]: latent space, shape {np.concatenate([result[4], result[5]]).shape}")

fig, ax = plt.subplots()
ax.set_title("Parity plot")
y_batch_test = np.array(result[2], dtype=float)
y_prob = np.array(result[3], dtype=float)
ax.scatter(y_batch_test, y_prob, color="blue", label=f"Predicted vs Actual (RMSE: {result[1]:.4f})")
min_val = min(min(y_batch_test), min(y_prob))
max_val = max(max(y_batch_test), max(y_prob))
ax.plot([min_val, max_val], [min_val, max_val], "r-")

ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.legend(loc="lower right")
