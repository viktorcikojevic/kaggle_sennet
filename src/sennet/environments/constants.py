from pathlib import Path
from sennet.environments.environments import DATA_DUMPS_DIR, DATA_DIR, MODEL_OUT_DIR


# example code for environments.py
# DATA_DIR = "/home/clay/research/kaggle/sennet/data"
# DATA_DUMPS_DIR = "/home/clay/research/kaggle/sennet/data_dumps"


REPO_DIR = Path(__file__).absolute().resolve().parent.parent.parent.parent


DATA_DIR = Path(DATA_DIR)
DATA_DUMPS_DIR = Path(DATA_DUMPS_DIR)
PROCESSED_DATA_DIR = DATA_DUMPS_DIR / "processed"
PROCESSED_2D_DATA_DIR = DATA_DUMPS_DIR / "processed_2d"
AUG_DUMP_DIR = DATA_DUMPS_DIR / "aug_dump"
MODEL_OUT_DIR = Path(MODEL_OUT_DIR)
TMP_SUB_MMAP_DIR = DATA_DUMPS_DIR / "tmp_mmaps"
PRETRAINED_DIR = DATA_DUMPS_DIR / "pretrained_checkpoints"
CONFIG_DIR = REPO_DIR / "configs"
STAGING_DIR = DATA_DUMPS_DIR / "staging"
FG_MASK_DIR = DATA_DIR.parent / "labeled_masks"


if __name__ == "__main__":
    pass
