# Commands

## Setup Envs
```bash
export DATASET_DIR="/opt/kaggle/sennet/data/blood-vessel-segmentation"
export DATA_DUMP_ROOT="/opt/kaggle/sennet/data_dumps"
export PROCESSED_DATA_DIR="$DATA_DUMP_ROOT/processed"
export MODEL_DIR="$DATA_DUMP_ROOT/models"
export NN_MODEL_OUT_DIR="$DATA_DUMP_ROOT/nn_predicted"
export MODE="train"
source source.bash
```

## Train Transformer Models
```bash
parallel --jobs 1 --eta --bar --progress \
  python3 tools/generate_mmap_dataset.py \
  --path {} \
  --output-dir "${PROCESSED_DATA_DIR}" \
  ::: $(ls -d "${DATASET_DIR}/${MODE}/"*)
```
