from sennet.environments.constants import STAGING_DIR, REPO_DIR, MODEL_OUT_DIR
from distutils.dir_util import copy_tree
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
import shlex
import json
import yaml
import os


CONFIG_PATH = Path(__file__).absolute().resolve().parent.parent / "configs" / "submission.yaml"


def main():
    with open(CONFIG_PATH, "rb") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    assert "KAGGLE_USERNAME" in os.environ, f"KAGGLE_USERNAME not found in env"
    assert "KAGGLE_KEY" in os.environ, f"KAGGLE_KEY not found in env"

    dataset_name = f"sumo-sennet-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(f"{dataset_name = }")

    model_names = cfg["predictors"]["models"]
    print(f"copying {len(model_names)} models")

    staging_path = STAGING_DIR / dataset_name
    staging_path.mkdir(exist_ok=True, parents=True)
    print(f"dataset will be staged at: {dataset_name = }")

    meta = dict(
        id=f"sirapoabchaikunsaeng/{dataset_name}",
        title=dataset_name,
        isPrivate=True,
        licenses=[dict(name="other")]
    )
    (staging_path / "dataset-metadata.json").write_text(json.dumps(meta))

    copy_tree(REPO_DIR / "src", str(staging_path / "src"))
    (staging_path / "tools").mkdir(exist_ok=True, parents=True)
    for p in Path(REPO_DIR / "tools").glob("*.py"):
        shutil.copy2(p, staging_path / "tools" / p.name)
    for model_name in model_names:
        model_path = MODEL_OUT_DIR / model_name
        copy_tree(model_path, str(staging_path / "data_dumps" / "models" / model_name))

    shutil.copy2(REPO_DIR / "source.bash", staging_path / "source.bash")
    copy_tree(REPO_DIR / "configs", str(staging_path / "configs"))

    (staging_path / "src" / "sennet" / "environments" / "environments.py").write_text(
        "DATA_DIR = '/kaggle/input/blood-vessel-segmentation'\n"
        + "DATA_DUMPS_DIR = '/tmp/data_dumps'\n"
        + f"MODEL_OUT_DIR = '/kaggle/input/{dataset_name}/data_dumps/models'"
    )

    command = f"kaggle datasets create -p \"{str(staging_path)}\" --dir-mode zip"
    print(f"running command: {command}")
    subprocess.call(shlex.split(command))


if __name__ == "__main__":
    main()
