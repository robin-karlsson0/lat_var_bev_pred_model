import torch
import torch.nn as nn

from lat_var_pred_model_prod import LatVarPredModelProd
from modules.unet_gan import UnetDecoder, UnetEncoder
from utils.lat_var_pred_aux import integrate_obs_and_lat_pred


class RoadAdvPredModelProd(nn.Module):

    def __init__(
        self,
        in_ch,
        in_size,
        out_ch,
        enc_dim,
        gen_enc_str,
        gen_dec_str,
        sample_type,
        lat_var_model_dict_pth,
        lat_var_model_ckpt_pth,
        road_thresh,
        **kwargs,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim

        self.sample_type = sample_type
        self.road_thresh = road_thresh

        ############
        #  Models
        ############
        # Latent variable predictive model
        with open(lat_var_model_dict_pth, 'rb') as f:
            model_config_dict = pickle.load(f)

        self.road_pred_model = LatVarPredModelProd(**model_config_dict)
        checkpoint = torch.load(lat_var_model_ckpt_pth)
        self.road_pred_model.load_state_dict(checkpoint['state_dict'])
        self.road_pred_model.eval()

        # Unet
        self.encoder = UnetEncoder(gen_enc_str, input_ch=self.in_ch)
        vec_dim = int(gen_enc_str.split(',')[-1].split('x')[-1])
        unet_out_ch = 4  # int(dec_str.split(',')[-1].split('x')[-1])
        self.decoder = UnetDecoder(gen_enc_str,
                                   gen_dec_str,
                                   vec_dim,
                                   output_ch=unet_out_ch,
                                   output_activation='sigmoid')

        if self.sample_type == 'all':
            self.unpack_sample = self.unpack_sample_all
        elif self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
        elif self.sample_type == 'nusc_road':
            self.unpack_sample = self.unpack_sample_nusc_road
        else:
            raise IOError(f'Undefined type ({type})')

    def unpack_sample_all(self, x):
        '''
        Args:
            type: String specifying how to unpack samples
                  'all' --> road semantic + lidar intensity
                  'road' --> road semantic only
        '''
        x_present = x[:, 0:2]
        x_future = x[:, 2:4]
        x_full = x[:, 4:6]
        x_traj = x[:, 6:7]

        x_target = x_full.clone()

        # Probabilistic value range (0, 1) --> (-1, +1)
        x_present[:, 0:1] = 2 * x_present[:, 0:1] - 1
        x_future[:, 0:1] = 2 * x_future[:, 0:1] - 1
        x_full[:, 0:1] = 2 * x_full[:, 0:1] - 1

        x_in = torch.concat([x_present, x_future])
        x_oracle = torch.concat([x_full, x_full])
        x_target = torch.concat([x_target, x_target])

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        mask = x_target[:, 0] > POS_THRESH
        x_target[mask, 0] = 1.
        mask = x_target[:, 0] < NEG_THRESH
        x_target[mask, 0] = 0.

        m_road = ~(x_target[:, 0] == 0.5)
        m_road[(x_target[:, 0] < POS_THRESH)
               & (x_target[:, 0] > NEG_THRESH)] = False

        m_intensity = m_road.clone()
        m_intensity[x_target[:, 0] < POS_THRESH] = False

        m_target = torch.stack([m_road, m_intensity], dim=1)

        return x_in, x_oracle, x_target, m_target, x_traj

    def unpack_sample_road(self, x):

        x_present = x[:, 0:1]
        x_future = x[:, 2:3]
        x_full = x[:, 4:5]
        x_traj = x[:, 6:7]

        x_target = x_full.clone()

        # Probabilistic value range (0, 1) --> (-1, +1)
        x_present[:, 0:1] = 2 * x_present[:, 0:1] - 1
        x_future[:, 0:1] = 2 * x_future[:, 0:1] - 1
        x_full[:, 0:1] = 2 * x_full[:, 0:1] - 1

        x_in = torch.concat([x_present, x_future])
        x_oracle = torch.concat([x_full, x_full])
        x_target = torch.concat([x_target, x_target])

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        mask = x_target[:, 0] > POS_THRESH
        x_target[mask, 0] = 1.
        mask = x_target[:, 0] < NEG_THRESH
        x_target[mask, 0] = 0.

        m_road = ~(x_target[:, 0] == 0.5)
        m_road[(x_target[:, 0] < POS_THRESH)
               & (x_target[:, 0] > NEG_THRESH)] = False

        m_target = m_road.unsqueeze(1)

        return x_in, x_oracle, x_target, m_target, x_traj

    def unpack_sample_nusc_road(self, x):
        '''
        Tensor (19,256,256)
            [0]:  road_present,      <--
            [1]:  intensity_present,
            [2]:  rgb_present[0],
            [3]:  rgb_present[1],
            [4]:  rgb_present[2],
            [5]:  elevation_present,
            [6]:  road_future,       <--
            [7]:  intensity_future,
            [8]:  rgb_future[0],
            [9]:  rgb_future[1],
            [10]: rgb_future[2],
            [11]: elevation_future,
            [12]: road_full,         <--
            [13]: intensity_full,
            [14]: rgb_full[0],
            [15]: rgb_full[1],
            [16]: rgb_full[2],
            [17]: elevation_full,
            [18]: traj_label,

        Returns:
            x_in: Input observation tensor in range (-1, 1)
            x_oracle: Full observation tensor in range(-1, 1)
            x_target: Thresholded full observation tensor in range (0, 1)
            m_target: Mask of thresholded elements to use in objective
        '''
        x_present = x[:, 0:5]
        x_future = x[:, 6:11]
        x_full = x[:, 12:17]
        x_traj = x[:, 18:19]

        # (road, RGB, elevation)
        # x_present = torch.concat((x[:, 0:1], x[:, 2:5], x[:, 5:6]), dim=1)
        # x_future = torch.concat((x[:, 6:7], x[:, 8:11], x[:, 11:12]), dim=1)
        # x_full = torch.concat((x[:, 12:13], x[:, 14:17], x[:, 17:18]), dim=1)

        x_target = x_full.clone()

        # Probabilistic value range (0, 1) --> (-1, +1)
        x_present[:, 0:1] = 2 * x_present[:, 0:1] - 1
        x_future[:, 0:1] = 2 * x_future[:, 0:1] - 1
        x_full[:, 0:1] = 2 * x_full[:, 0:1] - 1

        x_in = torch.concat([x_present, x_future])
        x_oracle = x_full
        x_target = torch.concat([x_target, x_target])

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        mask = x_target[:, 0] > POS_THRESH
        x_target[mask, 0] = 1.
        mask = x_target[:, 0] < NEG_THRESH
        x_target[mask, 0] = 0.

        m_road = ~(x_target[:, 0] == 0.5)
        m_road[(x_target[:, 0] < POS_THRESH)
               & (x_target[:, 0] > NEG_THRESH)] = False

        # All non-zero intensities assumed True observations
        m_intensity = x_target[:, 1] > 0

        # All non-black RGB values assumed True observations
        m_rgb = torch.sum(x_target[:, 2:], dim=1) > 0

        # (B, 5, 256, 256)
        m_target = torch.stack([m_road, m_intensity, m_rgb], dim=1)

        return x_in, x_oracle, x_target, m_target, x_traj

    @staticmethod
    def make_nonroad_intensity_zero(intensity, road, thres=0):
        # is_road_mask = (road > thres)
        # intensity[~is_road_mask] = 0
        intensity[road == 0] = 0

        return intensity

    def forward(self, x_in, m_target):
        '''
        Model:
            x --> enc() --> h --> fc() --> h' --> dec() --> y

        Args:
            x: (B, 2, H, W)
                [0]: Road
                [1]: Intensity
        '''
        x_pred = x_in.clone()
        #########################
        #  Representations
        #    1. Obs. road
        #    2. Obs. intensity
        #########################
        x_road = x_in[:, 0:1]  # Confirm if correct

        m_road, m_int, m_rgb = m_target.chunk(3, dim=1)

        # m_road = m_target[:, 0:1]
        # m_int = m_target[:, 1:2]
        # m_rgb = m_target[:, 2:5]
        # x_road, x_int = x_in.chunk(2, dim=1)
        # m_road, m_int = m_target.chunk(2, dim=1)

        ##########################
        #  Structure prediction
        ##########################
        with torch.no_grad():
            x_pred_road = self.road_pred_model.forward(x_road)[0]
        mode_idx = 0  # random.randint(0, self.road_pred_model.y_dim - 1)
        x_pred_road = x_pred_road[:, mode_idx]

        ########################
        #  Texture prediction
        ########################
        # Replace 'observed' road with 'obs + pred' road
        x_pred_road = integrate_obs_and_lat_pred(x_road,
                                                 x_pred_road,
                                                 m_road,
                                                 threshold=self.road_thresh)
        x_pred[:, 0:1] = x_pred_road

        x_pred_adv = x_pred.clone()

        # Remove non-road intensity values
        x_pred_adv[:, 1:2][~x_pred_road.bool()] = 0
        # Remove non-road RGB values
        x_pred_adv[:, 2:5][torch.tile(~x_pred_road.bool(), (1, 3, 1, 1))] = 0

        # Encoder-decoder model
        with torch.no_grad():
            h, enc_outs = self.encoder(x_pred_adv)
            x_pred_int_rgb = self.decoder(h, enc_outs)

        # Finally override predictions with observations
        x_pred[:, 0:1] = integrate_obs_and_lat_pred(x_road,
                                                    x_pred[:, 0:1],
                                                    m_road,
                                                    threshold=self.road_thresh,
                                                    override_pred=True)

        # Remove non-road intensity values
        x_pred[:, 1:2][~x_pred[:, 0:1].bool()] = 0
        # Remove non-road RGB values
        x_pred[:, 2:5][torch.tile(~x_pred[:, 0:1].bool(), (1, 3, 1, 1))] = 0

        # Integrate oberved and generated intensity
        m_int_fake = torch.logical_and(x_pred[:, 0:1], ~m_int)
        m_rgb_fake = torch.logical_and(x_pred[:, 0:1], ~m_rgb)

        # Replace False elements with generated elements
        x_pred[:, 1:2][m_int_fake] = x_pred_int_rgb[:, 0:1][m_int_fake]
        m_rgb_fake_tiled = torch.tile(m_rgb_fake, (1, 3, 1, 1))
        x_pred[:, 2:][m_rgb_fake_tiled] = x_pred_int_rgb[:,
                                                         1:][m_rgb_fake_tiled]

        return x_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('RoadAdvPredModel')
        parser.add_argument('--in_ch', type=int, default=2)
        parser.add_argument('--in_size', type=int, default=128)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=256,
                            help='Encoder output dim')
        parser.add_argument('--gen_enc_str',
                            default='2x32,2x64,2x128,2x256,2x256,2x512,2x256')
        #                             128   64    32    16     8     4     4
        parser.add_argument('--gen_dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32')
        #                                4     8    16    32   64  128
        parser.add_argument('--sample_type', type=str, default='all')
        parser.add_argument('--lat_var_model_dict_pth', type=str)
        parser.add_argument('--lat_var_model_ckpt_pth', type=str)
        parser.add_argument('--road_thresh', type=float, default=0.4999)

        return parent_parser


if __name__ == '__main__':
    import os
    import pickle
    from argparse import ArgumentParser

    import matplotlib as mpl
    mpl.use('agg')  # Must be before pyplot import
    import matplotlib.pyplot as plt

    from datamodules.bev_datamodule import (BEVDataModule,
                                            write_compressed_pickle)
    from utils.lat_var_pred_aux import integrate_obs_and_lat_pred

    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prob_conditioning', type=float, default=0)
    # Add model speficic args
    parser = RoadAdvPredModelProd.add_model_specific_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = RoadAdvPredModelProd(**dict_args)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()

    bev = BEVDataModule(
        train_data_dir=args.data_dir,
        val_data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_preproc=True,
        num_workers=1,
    )
    dataloader = bev.val_dataloader(shuffle=True)

    num_samples = len(bev.bev_dataset_train)

    bev_idx = 0
    subdir_idx = 0

    for idx, batch in enumerate(dataloader):

        x, _ = batch

        x_in, x_oracle, x_target, m_target, x_traj = model.unpack_sample(x)
        # Remove future sample
        x_in = x_in[0:args.batch_size]
        x_oracle = x_oracle[0:args.batch_size]
        m_target = m_target[0:args.batch_size]

        x_oracle = x_oracle.cuda()
        m_target = m_target.cuda()

        # x: (B,2,H,W)
        # m: (B,2,H,W)
        x_pred = model.forward(x_oracle, m_target)
        x_pred = x_pred.cpu()

        if bev_idx >= 1000:
            bev_idx = 0
            subdir_idx += 1
        filename = f'c_bev_{bev_idx}.pkl'
        output_path = f'./{args.save_dir}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        x_in[:, 1:2] = 2 * x_in[:, 1:2] - 1
        x_in[:,
             1:2] = model.make_nonroad_intensity_zero(x_in[:, 1:2], x_in[:,
                                                                         0:1])

        if torch.rand(1).item() < args.prob_conditioning:
            mask = x_traj == 1
            x_in[:, 0:1][mask] = 1

        # x = (1,11)
        #     [0] : in_road (possibly w. conditioning) [-1,1]
        #     [1] : in_int [-1,1]
        #     [2:5] : in_rgb [0,1] <-- !
        #     [5] : compl_road [0,1]
        #     [6] : compl_int [0,1]
        #     [7:10] : compl_rgb [0,1]
        #     [10] : full_traj [0,1]
        x = torch.concat([x_in, x_pred, x_traj], dim=1)

        write_compressed_pickle(x, filename, output_path)

        x_in = x_in.cpu().numpy()
        x_pred = x_pred.cpu().numpy()

        num_rows = 2
        num_cols = 4

        size_per_fig = 6
        _ = plt.figure(figsize=(size_per_fig * num_cols,
                                size_per_fig * num_rows))

        # 1 In: Road
        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(x[0, 0].numpy())
        # 2 In: Intensity
        plt.subplot(num_rows, num_cols, 2)
        plt.imshow(x[0, 1].numpy())
        # 3 In: RGB
        plt.subplot(num_rows, num_cols, 3)
        plt.imshow(x[0, 2:5].numpy().transpose((1, 2, 0)))
        # 4 In: Traj
        plt.subplot(num_rows, num_cols, 4)
        plt.imshow(x[0, 10].numpy())
        # 5 Out: Road
        plt.subplot(num_rows, num_cols, 5)
        plt.imshow(x[0, 5].numpy())
        # 6 Out: Intensity
        plt.subplot(num_rows, num_cols, 6)
        plt.imshow(x[0, 6].numpy())
        # 7 Out: RGB
        plt.subplot(num_rows, num_cols, 7)
        plt.imshow(x[0, 7:10].numpy().transpose((1, 2, 0)))

        plt.tight_layout()
        plt.savefig(output_path + f'fig_{bev_idx}.png')
        plt.clf()
        plt.close()

        bev_idx += 1

        if idx % 10 == 0:
            print(f'idx {idx} / {num_samples} ({idx/num_samples*100:.2f}%)')
