from fm4m.path_utils import get_path_from_root

DATA_DIR = get_path_from_root("fm4m", "../data")
MODELS_PATH = get_path_from_root("fm4m", "./models")

MODEL_ALIASES = {
    "MHG-GED": "mhg",
    "SELFIES-TED": "bart",
    "MolFormer": "mol-xl",
    "Molformer": "mol-xl",
    "SMI-TED": "smi-ted",
}

MORDRED_MODEL = "Mordred"
MOL_XL_MODEL = "mol-xl"
BART_MODEL = "bart"
MHG_MODEL = "mhg"
SMI_TED_MODEL = "smi-ted"
MORGAN_FINGERPRINT = "MorganFingerprint"
