import os
from argparse import ArgumentParser

if __name__ == "__main__":
    '''
    Script for creating 'train' and 'val' dataset splits.

    How to use:
        python utils/split_dataset.py
            dataset_abs_path
            train_abs_path
            val_abs_path 
            100
            10 11 21 22 33 34 ...
    '''

    parser = ArgumentParser()
    parser.add_argument('dataset_abs_path', type=str)
    parser.add_argument('train_dir_abs_path', type=str)
    parser.add_argument('val_dir_abs_path', type=str)
    parser.add_argument('num_subdirs', type=int)
    parser.add_argument('val_subdir_idxs', type=int, nargs='+')
    args = parser.parse_args()

    root = args.dataset_abs_path
    train_root = args.train_dir_abs_path
    val_root = args.val_dir_abs_path
    num_subdirs = args.num_subdirs
    val_subdir_idxs = args.val_subdir_idxs

    train_subdirs = []
    val_subdirs = []

    # Partition subdirs into 'train' and 'val' sets
    for subdir_idx in range(num_subdirs):
        subdir_path = os.path.join(root, f'subdir{str(subdir_idx).zfill(3)}')
        if subdir_idx in val_subdir_idxs:
            val_subdirs.append(subdir_path)
        else:
            train_subdirs.append(subdir_path)

    # Create symbolically linked 'train' set
    if os.path.isdir(train_root) is False:
        os.makedirs(train_root)
    for idx, subdir_path in enumerate(train_subdirs):
        new_subdir_path = os.path.join(train_root,
                                       f'subdir{str(idx).zfill(3)}')
        os.system(f'ln -s {subdir_path} {new_subdir_path}')

    # Create symbolically linked 'val' set
    if os.path.isdir(val_root) is False:
        os.makedirs(val_root)
    for idx, subdir_path in enumerate(val_subdirs):
        new_subdir_path = os.path.join(val_root, f'subdir{str(idx).zfill(3)}')
        os.system(f'ln -s {subdir_path} {new_subdir_path}')
