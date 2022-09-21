import glob
import pickle

import matplotlib as mpl
# mpl.use('agg')  # Must be before pyplot import to avoid memory leak
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
                                                     resnet18_encoder)

from modules.large_mnist_decoder import LargeMNISTExpDecoder
from modules.large_mnist_encoder import LargeMNISTExpEncoder
from modules.residual_decoder import ResidualDecoder
from modules.residual_encoder import ResidualEncoder
from modules.unet import UnetDecoder, UnetEncoder

# from utils.visualizations import viz_mixture_preds, viz_preds


class LatVarPredModel(pl.LightningModule):
    '''
    Ref: https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    '''

    def __init__(
        self,
        batch_size,
        in_ch,
        in_size,
        out_ch,
        enc_dim,
        lat_dim,
        z_hidden_dim,
        lr,
        weight_decay,
        encoder_conv_chs,
        decoder_conv_chs,
        num_viz_samples,
        num_log_p_samples,
        test_sample_repeat,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.batch_size = batch_size
        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.lat_dim = lat_dim
        self.z_hidden_dim = z_hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder_conv_chs = encoder_conv_chs
        self.decoder_conv_chs = decoder_conv_chs
        self.num_viz_samples = num_viz_samples

        self.num_log_p_samples = num_log_p_samples
        self.prec_rec_samples = 10000
        self.unknown_viz_val = 0.5

        # Models
        self.encoder = LargeMNISTExpEncoder(self.in_ch, self.enc_dim,
                                            self.in_size,
                                            self.encoder_conv_chs)
        self.decoder = LargeMNISTExpDecoder(self.out_ch, self.enc_dim,
                                            self.enc_dim + self.lat_dim,
                                            self.in_size,
                                            self.decoder_conv_chs)
        # middle_ch_ratio = 0.25
        # enc_str = '4x16,4x32,4x64,4x128,4x128,4x256,4x256'
        # self.encoder = ResidualEncoder(self.in_ch, enc_dim, in_size,
        #                                middle_ch_ratio, enc_str)
        # dec_str = '4x256,4x256,4x256,4x128,4x64,4x32,4x16'
        # self.decoder = ResidualDecoder(self.enc_dim + self.lat_dim, out_ch,
        #                                middle_ch_ratio, dec_str)
        # enc_str = f'2x32,2x64,2x128,2x256,2x256,2x512,2x{self.enc_dim}'
        #           128   64    32    16     8     4     4
        # self.encoder = UnetEncoder(enc_str)
        # dec_str = '2x512,2x256,2x256,2x128,2x64,2x32'
        #                4     8    16    32   64  128
        # feat_map_flatten_dim = self.enc_dim * 2**2
        # self.fc_decoder = nn.Sequential(
        #     nn.Linear(self.enc_dim + self.lat_dim, self.enc_dim, bias=False),
        #     nn.BatchNorm1d(self.enc_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.enc_dim, feat_map_flatten_dim, bias=False),
        #     nn.BatchNorm1d(feat_map_flatten_dim),
        #     nn.LeakyReLU(),
        # )
        # self.decoder = UnetDecoder(dec_str, self.enc_dim + self.lat_dim)

        self.inference_fc = torch.nn.Sequential(
            nn.Linear(self.enc_dim, self.z_hidden_dim),
            nn.LeakyReLU(),
        )
        self.inference_mu = nn.Linear(self.z_hidden_dim, self.lat_dim)
        self.infernece_log_var = nn.Linear(self.z_hidden_dim, self.lat_dim)

        # Posterior q(z|x)
        self.fc_mu = nn.Linear(self.enc_dim, self.lat_dim)
        self.fc_log_var = nn.Linear(self.enc_dim, self.lat_dim)

        # Print input output layer dimensions
        self.example_input_array = torch.rand(
            (32, self.in_ch, self.in_size, self.in_size))

        # Static test samples
        # self.test_sample_paths = sorted(
        #     glob.glob('./test_samples/*masked.pkl'))
        # if len(self.test_sample_paths) == 0:
        #     raise IOError('No test samples found in \'./test_samples/\'')
        # self.test_sample_repeat = test_sample_repeat

        # print("\n\nNOTE Copies val sample !!!\n\n")
        # self.val_sample = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=self.lr,
        #                             weight_decay=self.weight_decay)
        return optimizer

    def viz_test_samples(self, device, num_samples):
        '''
        Visualize predictions on a set of test samples M times to qualititively
        evaluate diverstity.
        '''
        x = []
        m = []
        for sample_path in self.test_sample_paths:
            f = open(sample_path, 'rb')
            x_, m_ = pickle.load(f)
            m_ = torch.tensor(m_, dtype=torch.bool)
            x.append(x_)
            m.append(m_)
        x = torch.stack(x)
        m = torch.stack(m)
        x = x.unsqueeze(1)  # (B,1,H,W)
        m = m.unsqueeze(1)  # (B,1,H,W)

        x = x.to(device)
        m = m.to(device)

        x_hats = []
        for _ in range(num_samples):
            x_hat, _, _, _ = self.vae(x)
            x_hats.append(x_hat)

        x_viz = self.unnormalize(x)
        x_viz[~m] = 0.5
        B = x_viz.shape[0]

        viz = viz_preds(x_hats[0], x_viz, B)
        for idx in range(1, num_samples):
            viz_ = viz_preds(x_hats[idx], x_viz, B)
            viz = np.concatenate((viz, viz_[32:]))

        size_per_fig = 2
        plt.figure(figsize=((B * size_per_fig,
                             (num_samples + 1) * size_per_fig)))
        plt.imshow(viz)
        plt.tight_layout()

        self.logger.experiment.add_figure(f'viz_test',
                                          plt.gcf(),
                                          global_step=self.current_epoch)

    def binary_cross_entropy(self, x_hat, x, obs_mask=None, eps=1e-12):
        log_pxz = x * torch.log(x_hat +
                                eps) + (1. - x) * torch.log(1. - x_hat + eps)

        if obs_mask is not None:
            log_pxz = obs_mask * log_pxz

        return log_pxz.sum(dim=(1, 2, 3)) / obs_mask.sum(dim=(1, 2, 3))

    def kl_divergence(self, p_mu, p_std, q_mu, q_std):
        '''
        D_KL( p(x) || q(x) ) ==> Optimize q(x) to cover p(x)
        '''
        return torch.log(q_std) - torch.log(
            p_std) + (p_std**2 + (p_mu - q_mu)**2) / (2 * q_std**2) - 0.5

    def qzh(self, h):
        z_hidden = self.inference_fc(h)
        z_mu, z_log_var = self.fc_mu(z_hidden), self.fc_log_var(z_hidden)
        z_std = torch.exp(z_log_var / 2)

        return z_mu, z_std

    def forward(self, x_in):
        '''
                                 Latent distribution encode stochasticity
        x_{t} --> enc() --> h--> q(z|h) --> N(mu, std) --> z_{t+1}
                            |                                |
                            ├--------------------------------┘
                            |
                            v
                           h_lat --> dec() --> x_{t+1}
        '''

        h = self.encoder(x_in)  # MNIST
        # out_res = 1  # ResEnc
        # h = h[out_res][:, :, 0, 0]  # ResEnc
        # h, _ = self.encoder(x_in)  # Unet
        # h = h.flatten(start_dim=1)

        z_mu, z_std = self.qzh(h)

        q = torch.distributions.Normal(z_mu, z_std)
        z = q.rsample()

        enc_vec = torch.cat((h, z), dim=-1)

        # enc_vec = self.fc_decoder(enc_vec)
        # feat_map = enc_vec.reshape(-1, self.env_dim + self.lat_dim, 2, 2)

        # enc_vec = enc_vec.unsqueeze(-1).unsqueeze(-1)  # ResEnc
        x_hat = self.decoder(enc_vec)
        # out_res = 128  # ResEnc
        # x_hat = x_hat[out_res]  # ResEnc

        return x_hat

    def training_step(self, batch, batch_idx):
        '''
                                    Latent distribution encode stochasticity
        x_{t} ----> enc() --> h --> q(z|h) --> N(mu, std)
                                                   |
                                                   ├--> KL()
                                                   |
        x_{t+1} --> enc() --> h --> q(z|h) --> N(mu, std) --> z_{t+1}
                                                                  |
                                                                  |
        x_{t} --> enc() --> h--> h_lat <--------------------------┘
                                   |
                                   └--> dec() --> x_{t+1}
        '''
        x, _ = batch

        # self.val_sample = x

        ###############
        #  Input x:s
        ###############
        x_1_prob = x[:, 0:1]  # Extract 'road_present' tensors
        x_2_prob = x[:, 1:2]  # Extract 'road_future' tensors
        x_3_prob = x[:, 2:3]  # Extract 'road_full' tensors

        # Value range (0, 1) --> (-1, +1)
        x_1 = 2 * x_1_prob - 1
        x_2 = 2 * x_2_prob - 1
        x_3 = 2 * x_3_prob - 1

        x_in = torch.concat([x_1, x_2])
        x_oracle = torch.concat([x_3, x_3])
        x_target = torch.concat([x_3_prob, x_3_prob])

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        x_target[x_target > POS_THRESH] = 1.
        x_target[x_target < NEG_THRESH] = 0.

        m_pred = ~(x_target == 0.5)
        m_pred[(x_target < POS_THRESH) & (x_target > NEG_THRESH)] = False

        ######################
        #  Encoding vectors
        ######################
        x_in_w_oracle = torch.cat([x_in, x_oracle])
        h = self.encoder(x_in_w_oracle)
        # out_res = 1  # ResEnc
        # h = h[out_res][:, :, 0, 0]  # ResEnc
        h, h_oracle = h.chunk(2)

        ##########################
        #  Latent distributions
        ##########################
        z_mu, z_std = self.qzh(h)
        z_mu_oracle, z_std_oracle = self.qzh(h_oracle)

        ###############################
        #  Distribution optimization
        ###############################

        # Optimize q(x) = N_{t}(z) to cover p(x) = N_{t+1}(z)
        kl = self.kl_divergence(z_mu_oracle, z_std_oracle, z_mu, z_std)
        kl = kl.sum(dim=-1)  # (N)

        # Optimize q(x_1) = q(x_2)
        z_mu_1, z_mu_2 = z_mu.chunk(2)
        z_std_1, z_std_2 = z_std.chunk(2)
        kl_1 = self.kl_divergence(z_mu_1, z_std_1, z_mu_2, z_std_2)
        kl_2 = self.kl_divergence(z_mu_2, z_std_2, z_mu_1, z_std_1)
        js = 0.5 * kl_1 + 0.5 * kl_2
        js = js.sum(dim=-1)  # (N)

        ################
        #  Prediction
        ################
        # Encoding h_{t} with oracle z_{t+1}
        q = torch.distributions.Normal(z_mu_oracle, z_std_oracle)
        z_oracle = q.rsample()
        h_lat = torch.cat((h, z_oracle), dim=-1)

        # h_lat = h_lat.unsqueeze(-1).unsqueeze(-1)  # ResEnc
        x_hat = self.decoder(h_lat)
        # out_res = 128  # ResEnc
        # x_hat = x_hat[out_res]  # ResEnc

        recon_loss = self.binary_cross_entropy(x_hat, x_target, m_pred)

        elbo = kl - recon_loss
        elbo = elbo.mean() + js.mean()

        self.log_dict({
            'train_elbo': elbo,
            'train_kl': kl.mean(),
            'train_js': js.mean(),
            'train_recon': recon_loss.mean(),
        })

        return elbo

    def validation_step(self, batch, batch_idx):

        self.encoder.train()
        self.inference_fc.train()
        self.fc_mu.train()
        self.fc_log_var.train()
        self.decoder.train()

        # if self.val_sample is not None:
        #     x = self.val_sample
        # else:
        #     x, _ = batch
        x, _ = batch

        # Remove full observation
        x_1_prob = x[:, 0:1]  # Extract 'road_present' tensors
        x_2_prob = x[:, 1:2]  # Extract 'road_future' tensors
        x_3_prob = x[:, 2:3]  # Extract 'road_full' tensors

        # Value range (0, 1) --> (-1, +1)
        x_1 = 2 * x_1_prob - 1
        x_2 = 2 * x_2_prob - 1

        x_in = torch.concat([x_1, x_2])

        x_target = torch.concat([x_3_prob, x_3_prob])

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        x_target[x_target > POS_THRESH] = 1.
        x_target[x_target < NEG_THRESH] = 0.

        m_pred = ~(x_target == 0.5)
        m_pred[(x_target < POS_THRESH) & (x_target > NEG_THRESH)] = False

        x_hat = self.forward(x_in)

        recon_loss = self.binary_cross_entropy(x_hat, x_target, m_pred)

        self.log('val_recon', recon_loss.mean())

        #        x, _ = batch
        #        x_prob = self.unnormalize(x)
        #        device = x.get_device()
        #
        #        x_hat, kl, _, z_std = self.vae(x_cat)
        #
        #        recon_loss = self.binary_cross_entropy(x_hat, x_prob, m_cat)
        #
        #        elbo = kl - recon_loss
        #        self.log('val_elbo', elbo.mean())
        #        self.log('val_recon', recon_loss.mean())
        #        self.log('val_kl', kl.mean())
        #
        if batch_idx == 0 and self.current_epoch % 1 == 0:
            #####################################
            #  Visualize input and predictions
            #####################################
            x_hats = []
            for _ in range(10):
                x_hat = self.forward(x_in)
                x_hats.append(x_hat)

            # Avoid overshooting batch size
            num_viz_samples = min(x.shape[0], self.num_viz_samples)
            rows = 1 + 10

            x_in_viz = x_in / 2 + 0.5
            viz = self.viz_mixture_preds(x_hats, x_in_viz, num_viz_samples)

            size_per_fig = 2
            plt.figure(figsize=((num_viz_samples * size_per_fig,
                                 rows * size_per_fig)))
            plt.imshow(viz, vmin=0, vmax=1)
            plt.tight_layout()

            for sample_idx in range(num_viz_samples):
                for k in range(10):
                    msg = str(k)
                    h_pos = sample_idx * self.in_size
                    v_pos = (k + 1) * self.in_size + 3
                    plt.text(h_pos, v_pos, msg, color='white')

            self.logger.experiment.add_figure(f'viz',
                                              plt.gcf(),
                                              global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        pass

    def viz_mixture_preds(self, x_hats: torch.Tensor, x: torch.Tensor,
                          num_viz_samples: int) -> np.array:
        '''
        Args:
            x_hats: List of K prediction batch tensor (B,C,H,W) in interval (0,1)
            x: Input batch tensor (B,C,H,W) in interval (0,1).
            num_viz_samples:

        Returns:
            Array of inputs and reconstructed predictions (K+1,N)
        '''
        B = x_hats[0].shape[0]
        K = len(x_hats)
        num_viz_samples = min(B, num_viz_samples)

        x_viz = x[:num_viz_samples]
        x_viz = self.batch_tensor_to_img_array(x_viz)

        x_hat_vizs = []
        for k in range(K):
            x_hat_viz = x_hats[k][:num_viz_samples]
            x_hat_viz = self.batch_tensor_to_img_array(x_hat_viz)
            x_hat_vizs.append(x_hat_viz)
        x_hat_vizs = np.concatenate(x_hat_vizs)

        viz = np.concatenate((x_viz, x_hat_vizs))

        return viz

    @staticmethod
    def batch_tensor_to_img_array(x: torch.Tensor) -> np.array:
        '''
        Args:
            x: Batch tensor (B,C,H,W) in interval (0,1).

        Returns:
            np.array (H,W*B)
        '''
        x = x.detach().cpu().numpy()
        batch_size, ch, _, _ = x.shape
        x = [x[idx] for idx in range(batch_size)]
        x = np.concatenate(x, axis=2)
        # Convert to grayscale or RGB
        if ch == 1:
            x = x[0]
        elif ch == 3:
            # x = np.transpose(x, (1,2,0))
            raise NotImplementedError()

        return x

    def validation_epoch_end(self, _):

        ############################
        #  Visualize test samples
        ############################
        # self.viz_test_samples(device, self.test_sample_repeat)
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LatVarPredModel')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--in_ch', type=int, default=1)
        parser.add_argument('--in_size', type=int, default=128)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=256,
                            help='Encoder output dim')
        parser.add_argument('--lat_dim', type=int, default=8)
        parser.add_argument(
            '--z_hidden_dim',
            type=int,
            default=256,
            help='Inference model output dim (before linear model: mu, std')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--encoder_conv_chs',
                            type=list,
                            default=[32, 64, 128, 256, 512, 128])
        parser.add_argument('--decoder_conv_chs',
                            type=list,
                            default=[64, 32, 16, 8, 4])
        parser.add_argument('--num_viz_samples', type=int, default=16)
        parser.add_argument('--num_log_p_samples', type=int, default=128)
        parser.add_argument('--test_sample_repeat', type=int, default=10)
        # parser.add_argument('', type=int, default=)
        return parent_parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    from datamodules.bev_datamodule import BEVDataModule

    parser = ArgumentParser()
    # Add program level args
    # Add model speficic args
    parser = LatVarPredModel.add_model_specific_args(parser)
    # Add all the vailable trainer option to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = LatVarPredModel(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)

    bev = BEVDataModule(
        data_dir=
        '/home/robin/projects/pc-accumulation-lib/bev_kitti360_128px_aug_gt_2',
        batch_size=args.batch_size,
        do_extrapolation=False,
        do_masking=False)

    trainer.fit(model, datamodule=bev)
