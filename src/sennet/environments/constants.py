from pathlib import Path
from .environments import DATA_DUMPS_DIR


# example code for environments.py
# DATA_DUMPS_DIR = "/home/clay/research/kaggle/sennet/data_dumps"


DATA_DUMPS_DIR = Path(DATA_DUMPS_DIR)
PROCESSED_DATA_DIR = DATA_DUMPS_DIR / "processed"
AUG_DUMP_DIR = DATA_DUMPS_DIR / "aug_dump"
MODEL_OUT_DIR = DATA_DUMPS_DIR / "models"
