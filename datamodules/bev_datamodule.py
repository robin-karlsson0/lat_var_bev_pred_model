import argparse
import copy
import glob
import gzip
import os
import pickle
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
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
        if not do_shuffle:
            self.sample_paths = sorted(self.sample_paths)

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

        self.min_elements = 0.01 * 256 * 256

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        '''
        Returns
            Tensor (7,256,256)
                [0]: road_present
                [1]: intensity_present
                [2]: road_future
                [3]: intensity_future
                [4]: road_full
                [5]: intensity_full
                [6]: traj_label
        '''
        while True:
            sample_path = self.sample_paths[idx]
            sample_path = os.path.join(self.abs_root_path, sample_path)

            sample = self.read_compressed_pickle(sample_path)

            num_obs_elem_present = np.sum(sample['road_present'] != 0.5)
            num_obs_elem_future = np.sum(sample['road_future'] != 0.5)
            num_obs_elem_full = np.sum(sample['road_full'] != 0.5)
            if (num_obs_elem_present < self.min_elements
                    or num_obs_elem_future < self.min_elements
                    or num_obs_elem_full < self.min_elements):
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

        # Make non-road intensity 0
        intensity_present = self.make_nonroad_intensity_zero(
            intensity_present, road_present)
        intensity_future = self.make_nonroad_intensity_zero(
            intensity_future, road_future)
        intensity_full = self.make_nonroad_intensity_zero(
            intensity_full, road_full)

        #################################
        #  Make dense trajectory label
        #################################
        traj_label = np.zeros((256, 256))
        for traj in sample['trajs_full']:
            traj = self.preproc_traj(traj)
            traj = self.draw_trajectory(traj, 256, 256, traj_width=1)
            mask = traj == 1
            traj_label[mask] = 1

        #######################################
        #  Make road under observed vehicles
        #######################################
        for traj in sample['trajs_present']:
            traj = self.preproc_traj(traj)
            traj = self.draw_trajectory(traj, 256, 256, traj_width=4)
            mask = traj == 1
            road_present[mask] = 1

        for traj in sample['trajs_future']:
            traj = self.preproc_traj(traj)
            traj = self.draw_trajectory(traj, 256, 256, traj_width=4)
            mask = traj == 1
            road_future[mask] = 1

        for traj in sample['trajs_full']:
            traj = self.preproc_traj(traj)
            traj = self.draw_trajectory(traj, 256, 256, traj_width=4)
            mask = traj == 1
            road_full[mask] = 1

        input_tensor = np.stack([
            road_present, intensity_present, road_future, intensity_future,
            road_full, intensity_full, traj_label
        ])
        input_tensor = torch.tensor(input_tensor, dtype=torch.float)

        # Random rotation
        if self.do_rotation:
            k = random.randrange(0, 4)
            input_tensor = torch.rot90(input_tensor, k, (-2, -1))

        return input_tensor, torch.tensor([0])

    def preproc_traj(self, traj: np.array):
        '''
        NOTE Need to make a copy to prevent overriding original trajectory.

        Args:
            traj: (N, 2) matrix with (i, j) coordinates.
        '''
        traj = copy.deepcopy(traj[:, 0:2])
        traj[:, 1] = 255 - traj[:, 1]
        # Convert to point list
        n = traj.shape[0]
        traj = [(int(traj[idx, 0]), int(traj[idx, 1])) for idx in range(n)]
        traj = self.remove_duplicate_pnts(traj)

        return traj

    @staticmethod
    def make_nonroad_intensity_zero(intensity, road, thres=0.5):
        # Make non-road intensity 0
        is_road_mask = (road > thres)
        intensity[~is_road_mask] = 0.
        return intensity

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

    def draw_trajectory(self,
                        traj: np.ndarray,
                        I: int,
                        J: int,
                        traj_width: int = 5):
        '''
        Args:
            traj: (N,2) matrix with (i, j) coordinates
        '''
        label = np.zeros((I, J))
        for idx in range(len(traj) - 1):

            pnt_0 = traj[idx]
            pnt_1 = traj[idx + 1]

            cv2.line(label, pnt_0, pnt_1, 1, traj_width)

        return label

    @staticmethod
    def remove_duplicate_pnts(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

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
        train_data_dir: str = "./",
        val_data_dir: str = "./",
        batch_size: int = 128,
        num_workers: int = 1,
        persistent_workers=True,
        do_rotation: bool = False,
        do_extrapolation=False,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
        use_preproc=False,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        if use_preproc:
            self.bev_dataset_train = PreprocBEVDataset(
                self.train_data_dir,
                do_rotation=do_rotation,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
            self.bev_dataset_val = PreprocBEVDataset(
                self.val_data_dir,
                do_shuffle=True,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
        else:
            self.bev_dataset_train = BEVDataset(
                self.train_data_dir,
                do_rotation=do_rotation,
                do_extrapolation=do_extrapolation,
                do_masking=do_masking,
                mask_p_min=mask_p_min,
                mask_p_max=mask_p_max,
            )
            self.bev_dataset_val = BEVDataset(
                self.val_data_dir,
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
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.bev_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Absolute path to dataset root (i.e. ../dataset/).')
    parser.add_argument('save_dir',
                        type=str,
                        help='Path to preprocessed sample save directory.')
    parser.add_argument('--do_rotation', action='store_true')
    parser.add_argument('--do_shuffle', action='store_true')
    args = parser.parse_args()

    #    import matplotlib.pyplot as plt

    batch_size = 1

    bev = BEVDataModule(args.dataset_path,
                        args.dataset_path,
                        batch_size,
                        do_rotation=args.do_rotation)
    dataloader = bev.train_dataloader(shuffle=args.do_shuffle)

    num_samples = len(bev.bev_dataset_train)

    bev_idx = 0
    subdir_idx = 0

    for idx, batch in enumerate(dataloader):

        input, _ = batch

        # Remove batch dim
        input = input[0]

        if bev_idx > 1000:
            bev_idx = 0
            subdir_idx += 1
        filename = f'bev_{bev_idx}.pkl'
        output_path = f'./{args.save_dir}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        write_compressed_pickle(input, filename, output_path)

        bev_idx += 1

        if idx % 100 == 0:
            print(f'idx {idx} / {num_samples} ({idx/num_samples*100:.2f}%)')
