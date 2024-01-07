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

2. run this to create numpy arrays from tiff files:

```bash
parallel --jobs 10 -k --lb --eta --bar --progress \
  python3 tools/generate_mmap_dataset.py \
  --path {} \
  --output-dir "${PROCESSED_DATA_DIR}" \
  ::: $(ls -d "${DATASET_DIR}/${MODE}/"*)

python3 tools/predict.py --out-dir "${DATA_DUMP_ROOT}/predicted" && \
python3 tools/pred_to_ply.py --path "${DATA_DUMP_ROOT}/predicted/ensembled/kidney_3_dense" && \
python3 tools/evaluate.py
```


3. Create rle labels

```bash
python tools/generate_rle_labels.py --path "${DATASET_DIR}/train_rles.csv"
```

---

# Installing Pyinterp
it's a bit of a headache because of this funny library called boost

## Installing Boost
1. download this and extract it https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
2. go inside the extracted dir, run the following:
```bash
sudo apt update
sudo apt install -y build-essential g++ cmake libeigen3-dev libboost-dev libgsl-dev python3-numpy 
./bootstrap.sh --prefix=/usr/
sudo ./b2 install
```
this should be enough, if things go wrong refer to this guy: https://stackoverflow.com/questions/12578499/how-to-install-boost-on-ubuntu

## Easy Part
```bash
pip3 install pyinterp
```
