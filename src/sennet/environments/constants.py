from pathlib import Path
from .environments import DATA_DUMPS_DIR, DATA_DIR


# example code for environments.py
# DATA_DIR = "/home/clay/research/kaggle/sennet/data"
# DATA_DUMPS_DIR = "/home/clay/research/kaggle/sennet/data_dumps"


DATA_DIR = Path(DATA_DIR)
DATA_DUMPS_DIR = Path(DATA_DUMPS_DIR)
PROCESSED_DATA_DIR = DATA_DUMPS_DIR / "processed"
PROCESSED_2D_DATA_DIR = DATA_DUMPS_DIR / "processed_2d"
AUG_DUMP_DIR = DATA_DUMPS_DIR / "aug_dump"
MODEL_OUT_DIR = DATA_DUMPS_DIR / "models"
TMP_SUB_MMAP_DIR = DATA_DUMPS_DIR / "tmp_mmaps"
PRETRAINED_DIR = DATA_DUMPS_DIR / "pretrained_checkpoints"
