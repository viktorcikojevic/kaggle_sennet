# Commands

## Setup Envs
```bash
export DATASET_DIR="/home/sennet"
export DATA_DUMP_ROOT="/home/data_dumps"
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
  --label-as-mask \
  --output-dir "${PROCESSED_DATA_DIR}" \
  ::: $(ls -d "${DATASET_DIR}/${MODE}/"*)

python3 tools/predict.py --out-dir "${DATA_DUMP_ROOT}/predicted" && \
python3 tools/pred_to_ply.py --path "${DATA_DUMP_ROOT}/predicted/ensembled/kidney_3_dense" && \
python3 tools/evaluate.py

# --- watershed stuffs ---
python3 tools/prepare_watershed_seed.py \
  --path "${DATA_DUMP_ROOT}/predicted/ensembled" \
  --skip-seed \
  --out-dir-name out
#python3 tools/prepare_watershed_seed.py \
#  --path "${DATA_DUMP_ROOT}/predicted/ensembled" \
#  --seed-threshold 0.9 \
#  --seed-dir-name seed \
#  --out-dir-name out

export PRED_ROOT="${DATA_DUMP_ROOT}/predicted/ensembled"
for i in $(ls "${PRED_ROOT}"); do
  echo "${PRED_ROOT}/${i}";
  /home/clay/research/kaggle/sennet/cpp/cmake-build-relwithdebinfo-clang/src/app/solveWatershed \
    --image="${PROCESSED_DATA_DIR}/${i}/image" \
    --pred="${PRED_ROOT}/${i}/chunk_00/mean_prob" \
    --seed="${PRED_ROOT}/${i}/chunk_00/thresholded_prob" \
    --out="${PRED_ROOT}/${i}/chunk_00/out" \
    --image-diff-threshold=4 \
    --label-upper-bound=0.2 \
    --label-lower-bound=0.001
  echo "---";
done

python3 tools/mmap_to_submission.py \
  --path "${PRED_ROOT}" \
  --folder-name "out" \
  --out-dir $PWD
```


3. Create rle labels

```bash
python tools/generate_rle_labels.py --path "${DATASET_DIR}/train_rles.csv"
```

---

```bash
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled_0005/kidney_2 --stride 2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled_0005/kidney_3_sparse --stride 2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled/kidney_2 --stride 2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled/kidney_3_sparse --stride 2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled_0005/kidney_2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled_0005/kidney_3_sparse && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled/kidney_2 && \
python3 tools/pred_to_ply_cc3d.py --path data_dumps/predicted/ensembled/kidney_3_sparse && \
echo "done :D"
```

```bash
--image /home/clay/research/kaggle/sennet/data_dumps/processed/kidney_3_sparse/image
--pred /home/clay/research/kaggle/sennet/data_dumps/predicted/for_rg/kidney_3_sparse/chunk_00/mean_prob
--seed /home/clay/research/kaggle/sennet/data_dumps/predicted/for_rg/kidney_3_sparse/chunk_00/seed
--out /home/clay/research/kaggle/sennet/data_dumps/predicted/for_rg/kidney_3_sparse/chunk_00/out
--image-diff-threshold 5
--label-upper-bound 0.2
--label-lower-bound 0.0001
```