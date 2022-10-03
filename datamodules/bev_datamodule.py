import copy
import glob
import gzip
import os
import pickle
import random

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import ndimage
from scipy.spatial import distance
from torch.utils.data import DataLoader, Dataset


class BEVDataset(Dataset):
    '''
    '''

    def __init__(
        self,
        abs_root_path,
        do_rotation=False,
        do_shuffle=False,
        do_extrapolation=False,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
    ):
        '''
        '''
        self.abs_root_path = abs_root_path
        self.sample_paths = glob.glob(
            os.path.join(self.abs_root_path, '*', '*.pkl.gz'))

        self.sample_paths = [
            os.path.relpath(path, self.abs_root_path)
            for path in self.sample_paths
        ]
        if do_shuffle:
            random.shuffle(self.sample_paths)

        self.num_samples = len(self.sample_paths)

        self.do_rotation = do_rotation
        self.do_extrapolation = do_extrapolation
        self.do_masking = do_masking
        self.mask_p_min = mask_p_min
        self.mask_p_max = mask_p_max

        # Road marking intensity transformation
        self.int_scaler = 20
        self.int_sep_scaler = 20
        self.int_mid_threshold = 0.5

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            sample_path = self.sample_paths[idx]
            sample_path = os.path.join(self.abs_root_path, sample_path)

            sample = self.read_compressed_pickle(sample_path)

            num_obs_elem_present = np.sum(sample['road_present'] != 0.5)
            num_obs_elem_future = np.sum(sample['road_future'] != 0.5)
            num_obs_elem_full = np.sum(sample['road_full'] != 0.5)
            if (num_obs_elem_present == 0 or num_obs_elem_future == 0
                    or num_obs_elem_full == 0):
                idx = self.get_random_sample_idx()
            else:
                break

        road_present = sample['road_present']
        road_future = sample['road_future']
        road_full = sample['road_full']
        road_present = road_present.astype(np.float32)
        road_future = road_future.astype(np.float32)
        road_full = road_full.astype(np.float32)

        intensity_present = sample['intensity_present']
        intensity_future = sample['intensity_future']
        intensity_full = sample['intensity_full']
        intensity_present = intensity_present.astype(np.float32)
        intensity_future = intensity_future.astype(np.float32)
        intensity_full = intensity_full.astype(np.float32)

        intensity_present = self.road_marking_transform(intensity_present)
        intensity_future = self.road_marking_transform(intensity_future)
        intensity_full = self.road_marking_transform(intensity_full)

        if self.do_extrapolation:
            pseudo_road_full, _, _ = self.gen_pseudo_bev(road_full)

            if self.do_masking:
                p = np.random.random()
                mask_prob = self.mask_p_max - (1. - self.mask_p_min) * p
                h, w = pseudo_road_full.shape
                mask = np.random.rand(h, w) < mask_prob
                pseudo_road_full[mask] = 0.5

            mask = ~(pseudo_road_full == 0.5)
            pseudo_road_full[mask] = pseudo_road_full[mask]

            mask = road_full == 0.5
            road_full[mask] = pseudo_road_full[mask]

        input_tensor = np.stack([
            road_present,
            intensity_present,
            road_future,
            intensity_future,
            road_full,
            intensity_full,
        ])
        input_tensor = torch.tensor(input_tensor, dtype=torch.float)

        # Random rotation
        if self.do_rotation:
            k = random.randrange(0, 4)
            input_tensor = torch.rot90(input_tensor, k, (-2, -1))

        return input_tensor, torch.tensor([0])

    def get_random_sample_idx(self):
        return np.random.randint(0, self.num_samples)

    def road_marking_transform(self, intensity_map):
        '''
        Args:
            intensity_map: Value interval (0, 1)
        '''
        intensity_map = self.int_scaler * self.sigmoid(
            self.int_sep_scaler * (intensity_map - self.int_mid_threshold))
        # Normalization
        intensity_map[intensity_map > 1.] = 1.
        return intensity_map

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def read_compressed_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)

    @staticmethod
    def distance_field(mat: np.array,
                       dist_from_pos: bool,
                       metric: str = 'chebyshev'):
        '''
        Assumes mat has a single semantic class with probablistic measurements
        p(x_{i,j}=True).
        '''
        if dist_from_pos:
            input = mat > 0.5
        else:
            input = mat < 0.5
        not_input = ~input

        input_inner = ndimage.binary_erosion(input)
        contour = input - input_inner.astype(int)

        dist_field = contour.astype(int)

        dist_field[not_input] = distance.cdist(
            np.argwhere(contour), np.argwhere(not_input), metric).min(0) + 1

        return dist_field

    def gen_pseudo_bev(self, bev, metric: str = 'sqeuclidean'):

        pos_dist_field = self.distance_field(bev, True, metric)
        neg_dist_field = self.distance_field(bev, False, metric)

        pseudo_bev = copy.deepcopy(bev)

        for i in range(128):
            for j in range(128):

                obs = bev[i, j]
                if obs > 0.5 or obs < 0.5:
                    continue

                pos_dist = pos_dist_field[i, j]
                neg_dist = neg_dist_field[i, j]

                if pos_dist < neg_dist:
                    pseudo_obs = 1.
                else:
                    pseudo_obs = 0.
                pseudo_bev[i, j] = pseudo_obs

        return pseudo_bev, pos_dist_field, neg_dist_field


class PreprocBEVDataset(BEVDataset):
    '''
    '''

    def __init__(
        self,
        abs_root_path,
        do_rotation=False,
        do_shuffle=False,
        do_extrapolation=False,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
    ):
        super().__init__(
            abs_root_path,
            do_rotation,
            do_shuffle,
            do_extrapolation,
            do_masking,
            mask_p_min,
            mask_p_max,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_path = self.sample_paths[idx]
        sample_path = os.path.join(self.abs_root_path, sample_path)
        sample = self.read_compressed_pickle(sample_path)

        # Random rotation
        if self.do_rotation:
            k = random.randrange(0, 4)
            sample = torch.rot90(sample, k, (-2, -1))

        return sample, torch.tensor([0])


class BEVDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 128,
        num_workers: int = 0,
        do_rotation: bool = False,
        do_extrapolation=False,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
        use_preproc=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if use_preproc:
            self.bev_dataset_train = PreprocBEVDataset(
                self.data_dir,
                do_rotation=do_rotation,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
            self.bev_dataset_val = PreprocBEVDataset(
                self.data_dir,
                do_shuffle=True,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
        else:
            self.bev_dataset_train = BEVDataset(
                self.data_dir,
                do_rotation=do_rotation,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
            self.bev_dataset_val = BEVDataset(
                self.data_dir,
                do_shuffle=True,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.bev_dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=True):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle=True):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )


def write_compressed_pickle(obj, filename, write_dir):
    '''Converts an object into byte representation and writes a compressed file.
    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    path = os.path.join(write_dir, f"{filename}.gz")
    pkl_obj = pickle.dumps(obj)
    try:
        with gzip.open(path, "wb") as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)


if __name__ == '__main__':
    '''
    For creating static set of test samples.
    '''

    #    import matplotlib.pyplot as plt

    batch_size = 1

    bev = BEVDataModule(
        '/home/robin/projects/pc-accumulation-lib/bev_kitti360_256px_aug_gt_3_rev',
        batch_size,
        do_rotation=False)
    dataloader = bev.train_dataloader(shuffle=False)

    num_samples = len(bev.bev_dataset_train)

    bev_idx = 0
    subdir_idx = 0
    savedir = 'bev_kitti360_256px_aug_gt_3_rev_preproc'

    for idx, batch in enumerate(dataloader):

        input, _ = batch

        # Remove batch dim
        input = input[0]

        if bev_idx > 1000:
            bev_idx = 0
            subdir_idx += 1
        filename = f'bev_{bev_idx}.pkl'
        output_path = f'./{savedir}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        write_compressed_pickle(input, filename, output_path)

        bev_idx += 1

        if idx % 100 == 0:
            print(f'idx {idx} / {num_samples} ({idx/num_samples*100:.2f}%)')

#        road_present = input[:, 0]
#        intensity_present = input[:, 1]
#        road_future = input[:, 2]
#        intensity_future = input[:, 3]
#        road_full = input[:, 4]
#        intensity_full = input[:, 5]
#
#        print(idx)
#        for batch_idx in range(batch_size):
#            # Present
#            plt.subplot(batch_size, 6, batch_idx * 6 + 1)
#            plt.imshow(road_present[batch_idx].numpy())
#            plt.subplot(batch_size, 6, batch_idx * 6 + 2)
#            plt.imshow(intensity_present[batch_idx].numpy())
#            # Future
#            plt.subplot(batch_size, 6, batch_idx * 6 + 3)
#            plt.imshow(road_future[batch_idx].numpy())
#            plt.subplot(batch_size, 6, batch_idx * 6 + 4)
#            plt.imshow(intensity_future[batch_idx].numpy())
#            # Full
#            plt.subplot(batch_size, 6, batch_idx * 6 + 5)
#            plt.imshow(road_full[batch_idx].numpy())
#            plt.subplot(batch_size, 6, batch_idx * 6 + 6)
#            plt.imshow(intensity_full[batch_idx].numpy())
#
#        plt.show()
