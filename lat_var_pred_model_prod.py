# import matplotlib as mpl

# mpl.use('agg')  # Must be before pyplot import to avoid memory leak
# import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import torch
import torch.nn as nn

# from modules.large_mnist_decoder import LargeMNISTExpDecoder
# from modules.large_mnist_encoder import LargeMNISTExpEncoder
from modules.unet import UnetDecoder, UnetEncoder


class LatVarPredModelProd(nn.Module):
    '''
    Ref: https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    '''

    def __init__(
        self,
        in_ch,
        in_size,
        out_ch,
        enc_dim,
        lat_dim,
        z_hidden_dim,
        y_dim,
        enc_str,
        dec_str,
        sample_type,
        **kwargs,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.lat_dim = lat_dim
        self.z_hidden_dim = z_hidden_dim
        self.y_dim = y_dim

        self.sample_type = sample_type

        ############
        #  Models
        ############
        # Unet
        self.encoder = UnetEncoder(enc_str, input_ch=self.in_ch)
        vec_dim = self.enc_dim + self.lat_dim
        unet_out_ch = 1  # int(dec_str.split(',')[-1].split('x')[-1])
        self.decoder = UnetDecoder(dec_str,
                                   vec_dim,
                                   input_ch=vec_dim,
                                   output_ch=unet_out_ch,
                                   output_activation='sigmoid')

        # Oracle q(z|x)
        self.inference_fc_oracle = torch.nn.Sequential(
            nn.Linear(self.enc_dim, self.z_hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu_oracle = nn.Linear(self.z_hidden_dim, self.lat_dim)
        self.fc_log_var_oracle = nn.Linear(self.z_hidden_dim, self.lat_dim)

        # Prior q(z|xh)
        self.inference_fc_prior = torch.nn.Sequential(
            nn.Linear(self.enc_dim + self.y_dim, self.z_hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu_prior = nn.Linear(self.z_hidden_dim, self.lat_dim)
        self.fc_log_var_prior = nn.Linear(self.z_hidden_dim, self.lat_dim)

        # Print input output layer dimensions
        self.example_input_array = torch.rand(
            (32, self.in_ch, self.in_size, self.in_size))

        if self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
        elif self.sample_type == 'nusc_road':
            self.unpack_sample = self.unpack_sample_nusc_road
        else:
            raise IOError(f'Undefined type ({type})')

    def qzh_oracle(self, h):
        z_hidden = self.inference_fc_oracle(h)
        z_mu = self.fc_mu_oracle(z_hidden)
        z_log_var = self.fc_log_var_oracle(z_hidden)
        z_std = torch.exp(z_log_var / 2)
        return z_mu, z_std

    def qzh_prior(self, hy):
        z_hidden = self.inference_fc_prior(hy)
        z_mu = self.fc_mu_prior(z_hidden)
        z_log_var = self.fc_log_var_prior(z_hidden)
        z_std = torch.exp(z_log_var / 2)
        return z_mu, z_std

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
        x_present = x[:, 0:1]
        x_future = x[:, 6:7]
        x_full = x[:, 12:13]

        # (road, RGB, elevation)
        # x_present = torch.concat((x[:, 0:1], x[:, 2:5], x[:, 5:6]), dim=1)
        # x_future = torch.concat((x[:, 6:7], x[:, 8:11], x[:, 11:12]), dim=1)
        # x_full = torch.concat((x[:, 12:13], x[:, 14:17], x[:, 17:18]), dim=1)

        x_target = x_full.clone()

        # Probabilistic value range (0, 1) --> (-1, +1)
        x_present[:, 0:1] = 2 * x_present[:, 0:1] - 1
        x_future[:, 0:1] = 2 * x_future[:, 0:1] - 1
        x_full[:, 0:1] = 2 * x_full[:, 0:1] - 1

        # Normalize RGB
        # x_present[:, 1:4] -= 0.5
        # x_present[:, 1:4] *= 2
        # x_future[:, 1:4] -= 0.5
        # x_future[:, 1:4] *= 2
        # x_full[:, 1:4] -= 0.5
        # x_full[:, 1:4] *= 2

        # Threshold and normalize elevation
        # elev = x_present[:, 4]
        # elev[elev < -2] = -2
        # elev[elev > 2] = 2
        # elev /= 2
        # x_present[:, 4] = elev
        # elev = x_future[:, 4]
        # elev[elev < -2] = -2
        # elev[elev > 2] = 2
        # elev /= 2
        # x_future[:, 4] = elev
        # elev = x_full[:, 4]
        # elev[elev < -2] = -2
        # elev[elev > 2] = 2
        # elev /= 2
        # x_full[:, 4] = elev

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

    def forward(self, x_in):
        '''
                                 Latent distribution encode stochasticity
        x_{t} --> enc() --> h--> q(z|h) --> N(mu, std) --> z_{t+1}
                            |                                |
                            ├--------------------------------┘
                            |
                            v
                           h_lat --> dec() --> x_{t+1}

        Args:
            x_in: (B, 1, H, W)

        Returns:
            x_hats: (B, K, 1, H, W)
        '''
        B = x_in.shape[0]
        device = x_in.get_device()

        h = self.encoder(x_in)

        ################
        #  Multimodal
        ################
        y = torch.eye(self.y_dim, device=device)  # (K, K)

        hs = []
        x_hats = []
        B = h.shape[0]
        z_mus = []
        z_stds = []
        for mixture_idx in range(1):

            y_ = y[:, mixture_idx:mixture_idx + 1]  # (K, 1)
            y_ = y_.T  # (1, K)
            y_ = torch.tile(y_, (B, 1))  # Y matrix (B, K)

            hy = torch.cat([h, y_], dim=1)  # (B, E+K)

            ##########################
            #  Latent distributions
            ##########################
            z_mu, z_std = self.qzh_prior(hy)

            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
            hs.append(h)
            z_mus.append(z_mu)
            z_stds.append(z_std)

            enc_vec = torch.cat((h, z), dim=-1)

            x_hat = self.decoder(enc_vec)
            # 'road' and 'intensity' output head pair
            # road_pred = self.road_head(out_feat)

            # x_hat = road_pred
            x_hats.append(x_hat)

        # Convert 'mixture ordered' list --> 'sample ordered' tensor
        x_hats = torch.stack(x_hats)  # (K, B, 1, H, W)
        x_hats = torch.transpose(x_hats, 1, 0)  # (B, K, 1, H, W)

        hs = torch.stack(hs)
        hs = torch.transpose(hs, 1, 0)

        z_mus = torch.stack(z_mus)
        z_mus = torch.transpose(z_mus, 1, 0)

        z_stds = torch.stack(z_stds)
        z_stds = torch.transpose(z_stds, 1, 0)

        return x_hats, hs, z_mus, z_stds

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LatVarPredModelProd')
        parser.add_argument('--in_ch', type=int, default=1)
        parser.add_argument('--in_size', type=int, default=256)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=128,
                            help='Encoder output dim')
        parser.add_argument('--lat_dim', type=int, default=32)
        parser.add_argument(
            '--z_hidden_dim',
            type=int,
            default=256,
            help='Inference model output dim (before linear model: mu, std')
        parser.add_argument('--y_dim', type=int, default=10)
        parser.add_argument(
            '--enc_str',
            default='2x16,2x32,2x64,2x128,2x256,2x256,2x512,2x128')
        parser.add_argument('--dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32,2x16')
        parser.add_argument('--sample_type', type=str, default='road')
        # parser.add_argument('', type=int, default=)
        return parent_parser


if __name__ == '__main__':
    # import os
    import pickle
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt

    # import onnx
    # import onnxruntime as ort
    from datamodules.bev_datamodule import BEVDataModule
    from utils.lat_var_pred_aux import integrate_obs_and_lat_pred

    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--checkpoint_path', type=str)
    # parser.add_argument('--make_onnx_model', action="store_true")
    parser.add_argument('--use_oracle_sample', action="store_true")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    # Add model speficic args
    parser = LatVarPredModelProd.add_model_specific_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = LatVarPredModelProd(**dict_args)

    with open('lat_var_pred_model_config.pkl', 'wb') as f:
        pickle.dump(dict_args, f)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()

    #######################
    #  Create ONNX model
    #######################
    # onnx_filename = f'lat_var_pred_model_bs{args.batch_size}.onnx'
    # if args.make_onnx_model:
    #     x = torch.randn(args.batch_size, 1, 256, 256, requires_grad=True)
    #     torch.onnx.export(
    #         model,
    #         x,
    #         onnx_filename,
    #         export_params=True,
    #         do_constant_folding=True,
    #         input_names=['input'],
    #         output_names=['output'],
    #     )
    # if os.path.isfile(onnx_filename) is not True:
    #     raise IOError(f'ONNX model does not exist ({onnx_filename})')
    # # Check that the model is well formed
    # onnx_model = onnx.load(onnx_filename)
    # onnx.checker.check_model(onnx_model)

    bev = BEVDataModule(
        train_data_dir=args.data_dir,
        val_data_dir=args.data_dir,
        batch_size=args.batch_size,
        do_rotation=True,
        do_extrapolation=False,
        do_masking=False,
        use_preproc=True,
        num_workers=1,
    )
    dataloader = bev.val_dataloader(shuffle=True)

    #####################
    #  Test ONNX model
    #####################
    # ort_session = ort.InferenceSession(onnx_filename,
    #                                    providers=['CUDAExecutionProvider'])

    for idx, batch in enumerate(dataloader):

        print(f'idx {idx}')

        x, _ = batch

        x_in, x_oracle, x_target, m_target = model.unpack_sample_road(x)
        if args.use_oracle_sample:
            x_in = x_oracle
        # Remove future sample
        x_in = x_in[0:args.batch_size]
        x_in = x_in.cuda()
        m_target = m_target[0:args.batch_size, 0]
        # x_oracle = x_oracle[0:1]
        # x_oracle = x_oracle.cuda()

        x_hats = []
        for _ in range(args.num_samples):

            # outputs = ort_session.run(None, {'input': x_in.numpy()})
            # x_hat = outputs[0]  # (B, K, 1, H, W)
            # x_hat = x_hat[0:1]

            x_hat, _, _, _ = model.forward(x_in)
            x_hats.append(x_hat)

        x_in = x_in.cpu().numpy()
        m_target = m_target.cpu().numpy()
        # x_oracle = x_oracle.cpu().numpy()
        x_hats = [x_hat.detach().cpu().numpy() for x_hat in x_hats]

        num_rows = args.num_samples
        num_cols = 1 + model.y_dim

        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(x_in[0, 0])
        # plt.imshow(x_oracle[0, 0])

        for sampling_idx in range(args.num_samples):
            for idx in range(1, model.y_dim):
                plt.subplot(num_rows, num_cols,
                            idx + 1 + sampling_idx * num_cols)
                x_pred = x_hats[sampling_idx][0, idx, 0]

                x_pred = integrate_obs_and_lat_pred(x_in[0, 0],
                                                    x_pred,
                                                    m_target[0],
                                                    threshold=0.4999,
                                                    is_np=True)
                plt.imshow(x_pred)

        plt.tight_layout()
        plt.show()
