from enum import Enum

from .mhg_model.load import load as mhg
from .selfies_ted import Selfies as Bart
from .smi_ted.smi_ted_light.load import load_smi_ted
from ..config.model_files import MHG_MODEL_PICKLE


class DownstreamModelType(Enum):
    XGBClassifier = "XGBClassifier"
    DefaultClassifier = "DefaultClassifier"
    SVR = "SVR"
    LinearRegression = "Linear Regression"
    KernelRidge = "Kernel Ridge"
    DefaultRegressor = "DefaultRegressor"


def load_smi_ted_model():
    return load_smi_ted()


def load_bart_model():
    model = Bart()
    model.load()
    return model


def load_mhg_model():
    return mhg.load(MHG_MODEL_PICKLE)
