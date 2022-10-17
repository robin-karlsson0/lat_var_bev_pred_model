# import pytorch_lightning as pl
import torch
import torch.nn as nn

# from det_var_pred_model_prod import DetPredModelProd
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
        **kwargs,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim

        self.sample_type = sample_type

        ############
        #  Models
        ############
        # Latent variable predictive model
        with open(lat_var_model_dict_pth, 'rb') as f:
            model_config_dict = pickle.load(f)

        # self.road_pred_model = DetPredModelProd(**model_config_dict)
        self.road_pred_model = LatVarPredModelProd(**model_config_dict)
        checkpoint = torch.load(lat_var_model_ckpt_pth)
        self.road_pred_model.load_state_dict(checkpoint['state_dict'])
        self.road_pred_model.eval()

        # Unet
        self.encoder = UnetEncoder(gen_enc_str, input_ch=self.in_ch)
        vec_dim = int(gen_enc_str.split(',')[-1].split('x')[-1])
        unet_out_ch = 1  # int(dec_str.split(',')[-1].split('x')[-1])
        self.decoder = UnetDecoder(gen_enc_str,
                                   gen_dec_str,
                                   vec_dim,
                                   output_ch=unet_out_ch,
                                   output_activation='sigmoid')

        if self.sample_type == 'all':
            self.unpack_sample = self.unpack_sample_all
        elif self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
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

        return x_in, x_oracle, x_target, m_target

    def unpack_sample_road(self, x):

        x_present = x[:, 0:1]
        x_future = x[:, 2:3]
        x_full = x[:, 4:5]

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

        return x_in, x_oracle, x_target, m_target

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
        x_road, x_int = x_in.chunk(2, dim=1)
        m_road, m_int = m_target.chunk(2, dim=1)

        ##########################
        #  Structure prediction
        ##########################
        with torch.no_grad():
            x_pred_road = self.road_pred_model.forward(x_road)[0]
        mode_idx = 0  # random.randint(0, self.road_pred_model.y_dim - 1)
        x_pred_road = x_pred_road[:, mode_idx]
        # Replace 'observed' road with 'obs + pred' road
        x_pred_road = integrate_obs_and_lat_pred(x_road, x_pred_road, m_road)
        x_pred[:, 0:1] = x_pred_road

        ########################
        #  Texture prediction
        ########################
        # Encoder-decoder model
        with torch.no_grad():
            h, enc_outs = self.encoder(x_pred)
            x_pred_int = self.decoder(h, enc_outs)

        # Integrate oberved and generated intensity
        m_fake = torch.logical_xor(x_pred_road, m_int)
        x_int[m_fake] = x_pred_int[m_fake]
        x_pred[:, 1:2] = x_int

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
        # parser.add_argument('', type=int, default=)
        return parent_parser


if __name__ == '__main__':
    import os
    import pickle
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt

    from datamodules.bev_datamodule import (BEVDataModule,
                                            write_compressed_pickle)
    from utils.lat_var_pred_aux import integrate_obs_and_lat_pred

    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--use_oracle_sample', action="store_true")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    # Add model speficic args
    parser = RoadAdvPredModelProd.add_model_specific_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = RoadAdvPredModelProd(**dict_args)

    # with open(args.dict_path, 'wb') as f:
    #     pickle.dump(dict_args, f)

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

        print(f'idx {idx}')

        x, _ = batch

        x_in, x_oracle, x_target, m_target = model.unpack_sample(x)
        if args.use_oracle_sample:
            x_in = x_oracle
        # Remove future sample
        x_in = x_in[0:args.batch_size]
        x_in = x_in.cuda()
        m_target = m_target[0:args.batch_size]
        m_target = m_target.cuda()

        x_pred = model.forward(x_in, m_target)

        if bev_idx >= 1000:
            bev_idx = 0
            subdir_idx += 1
        filename = f'c_bev_{bev_idx}.pkl'
        output_path = f'./{args.save_dir}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        x = torch.concat([x_in, x_pred], dim=1)

        write_compressed_pickle(x, filename, output_path)

        x_in = x_in.cpu().numpy()
        x_pred = x_pred.cpu().numpy()

        num_rows = args.batch_size
        num_cols = 4

        size_per_fig = 6
        _ = plt.figure(figsize=(size_per_fig * num_cols,
                                size_per_fig * num_rows))

        for idx in range(args.batch_size):
            plt.subplot(num_rows, num_cols, 1 + 0 + idx * num_cols)
            plt.imshow(x_in[idx, 0])
            plt.subplot(num_rows, num_cols, 1 + 1 + idx * num_cols)
            plt.imshow(x_in[idx, 1])
            plt.subplot(num_rows, num_cols, 1 + 2 + idx * num_cols)
            plt.imshow(x_pred[idx, 0])
            plt.subplot(num_rows, num_cols, 1 + 3 + idx * num_cols)
            plt.imshow(x_pred[idx, 1])

        plt.tight_layout()
        plt.savefig(output_path + f'fig_{bev_idx}.png')

        bev_idx += 1

        if idx % 100 == 0:
            print(f'idx {idx} / {num_samples} ({idx/num_samples*100:.2f}%)')
