from enum import Enum

from .mhg_model import load as mhg
from .selfies_ted import Selfies as Bart
from .smi_ted.smi_ted_light import load as smi_ted


class DownstreamModelType(Enum):
    XGBClassifier = "XGBClassifier"
    DefaultClassifier = "DefaultClassifier"
    SVR = "SVR"
    LinearRegression = "Linear Regression"
    KernelRidge = "Kernel Ridge"
    DefaultRegressor = "DefaultRegressor"


class ModelType(Enum):
    MORDRED_MODEL = "Mordred"
    MOL_XL_MODEL = "mol-xl"
    BART_MODEL = "bart"
    MHG_MODEL = "mhg"
    SMI_TED_MODEL = "smi-ted"
    MORGAN_FINGERPRINT = "MorganFingerprint"


def load_smi_ted_model():
    return smi_ted.load_smi_ted()


def load_bart_model():
    model = Bart()
    model.load()
    return model


def load_mhg_model():
    return mhg.load()
