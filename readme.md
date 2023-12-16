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



## Generate dataset

0. Note: copy imgs from kidney_3_sparse to kidney_3_dense, they're the same.

1. create a file `src/sennet/environments/environments.py` with the following content:

```python
DATA_DUMPS_DIR = "/home/viktor/Documents/kaggle/kaggle_sennet/data_dumps/" 
```
or wherever you want to store the data dumps.

2. run this to creat numpy arrays from tiff files:

```bash
parallel --jobs 10 -k --lb --eta --bar --progress \
  python3 tools/generate_mmap_dataset.py \
  --path {} \
  --output-dir "${PROCESSED_DATA_DIR}" \
  ::: $(ls -d "${DATASET_DIR}/${MODE}/"*)
```


3. Create rle labels

```bash
python tools/generate_rle_labels.py --path "${DATASET_DIR}/train_rles.csv"
```


## Generate debug dataset (optional) (not working yet!)


Create a debug dataset with 10 channels.
```bash
bash create_debug_dataset.sh
```