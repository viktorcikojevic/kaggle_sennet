
# Create debug dataset for training
export DATASET_DIR_DEBUG="/opt/kaggle/sennet/data/blood-vessel-segmentation-debug"

current_dir=$(pwd)

cd /opt/kaggle/sennet/data
cp -r blood-vessel-segmentation/ blood-vessel-segmentation-debug

cd blood-vessel-segmentation-debug/train
rm -r kidney_1_voi/ kidney_2/ kidney_3_dense/


cd kidney_1_dense/
cd images/
find . -type f ! \( -name '0000.tif' -o -name '0001.tif' -o -name '0002.tif' -o -name '0003.tif' -o -name '0004.tif' -o -name '0005.tif' -o -name '0006.tif' -o -name '0007.tif' -o -name '0008.tif' -o -name '0009.tif' \) -exec rm {} +
cd ..

cd labels
find . -type f ! \( -name '0000.tif' -o -name '0001.tif' -o -name '0002.tif' -o -name '0003.tif' -o -name '0004.tif' -o -name '0005.tif' -o -name '0006.tif' -o -name '0007.tif' -o -name '0008.tif' -o -name '0009.tif' \) -exec rm {} +
cd ..

cd ..


cd kidney_3_sparse/
cd images/
find . -type f ! \( -name '0000.tif' -o -name '0001.tif' -o -name '0002.tif' -o -name '0003.tif' -o -name '0004.tif' -o -name '0005.tif' -o -name '0006.tif' -o -name '0007.tif' -o -name '0008.tif' -o -name '0009.tif' \) -exec rm {} +
cd ..

cd labels
find . -type f ! \( -name '0000.tif' -o -name '0001.tif' -o -name '0002.tif' -o -name '0003.tif' -o -name '0004.tif' -o -name '0005.tif' -o -name '0006.tif' -o -name '0007.tif' -o -name '0008.tif' -o -name '0009.tif' \) -exec rm {} +
cd ..


cd $current_dir
rm -r data_dumps/*

parallel --jobs 10 -k --lb --eta --bar --progress \
  python3 tools/generate_mmap_dataset.py \
  --path {} \
  --output-dir "${PROCESSED_DATA_DIR}" \
  ::: $(ls -d "${DATASET_DIR_DEBUG}/${MODE}/"*)

python tools/generate_rle_labels.py --path "${DATASET_DIR_DEBUG}/train_rles.csv"