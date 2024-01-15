import os
import shutil
import numpy as np

def create_folds(base_dir, folder, n_folds=5):
    images_dir = os.path.join(base_dir, folder, 'images')
    labels_dir = os.path.join(base_dir, folder, 'labels')
    
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(image_files))
    
    fold_size = len(image_files) // n_folds
    for i in range(n_folds):
        fold_dir = os.path.join(base_dir, f'kidney_1_dense_fold_{i}')
        fold_images_dir = os.path.join(fold_dir, 'images')
        fold_labels_dir = os.path.join(fold_dir, 'labels')

        os.makedirs(fold_images_dir, exist_ok=True)
        os.makedirs(fold_labels_dir, exist_ok=True)

        test_indices = set(shuffled_indices[i * fold_size : (i + 1) * fold_size if i < n_folds - 1 else len(image_files)])

        for idx in range(len(image_files)):
            if idx not in test_indices:
                shutil.copy(os.path.join(images_dir, image_files[idx]), fold_images_dir)
                shutil.copy(os.path.join(labels_dir, label_files[idx]), fold_labels_dir)
		
        # get the list of files in the images and labels directory
        file_names = sorted(os.listdir(images_dir))
        # rename the files in the images and labels directory, format 04d
        new_file_names = [f'{i:04d}.tif' for i in range(len(file_names))]
        for file_name, new_file_name in zip(file_names, new_file_names):
            os.rename(os.path.join(images_dir, file_name), os.path.join(images_dir, new_file_name))
            os.rename(os.path.join(labels_dir, file_name), os.path.join(labels_dir, new_file_name))

  
base_dir = '/opt/kaggle/sennet/data/blood-vessel-segmentation/train/'
folder = 'kidney_1_dense'
create_folds(base_dir, folder)
