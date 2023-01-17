import matplotlib as mpl

mpl.use('agg')  # Must be before pyplot import to avoid memory leak
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.large_mnist_decoder import LargeMNISTExpDecoder
from modules.large_mnist_encoder import LargeMNISTExpEncoder
from modules.unet import UnetDecoder, UnetEncoder


class DetPredModel(pl.LightningModule):
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
        lr,
        beta,
        weight_decay,
        encoder_conv_chs,
        decoder_conv_chs,
        enc_str,
        dec_str,
        num_viz_samples,
        num_log_p_samples,
        test_sample_repeat,
        sample_type,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.batch_size = batch_size
        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.encoder_conv_chs = encoder_conv_chs
        self.decoder_conv_chs = decoder_conv_chs
        self.num_viz_samples = num_viz_samples

        self.num_log_p_samples = num_log_p_samples
        self.prec_rec_samples = 10000
        self.unknown_viz_val = 0.5
        self.sample_type = sample_type

        ############
        #  Models
        ############
        # MNIST
        # self.encoder = LargeMNISTExpEncoder(self.in_ch, self.enc_dim,
        #                                     self.in_size,
        #                                     self.encoder_conv_chs)
        # self.decoder = LargeMNISTExpDecoder(self.out_ch, self.enc_dim,
        #                                     self.enc_dim + self.lat_dim,
        #                                     self.in_size,
        #                                     self.decoder_conv_chs)
        # Unet
        self.encoder = UnetEncoder(enc_str, input_ch=self.in_ch)
        vec_dim = self.enc_dim
        unet_out_ch = 1  # int(dec_str.split(',')[-1].split('x')[-1])
        self.decoder = UnetDecoder(dec_str,
                                   vec_dim,
                                   input_ch=vec_dim,
                                   output_ch=unet_out_ch,
                                   output_activation='sigmoid')
        # output_activation='leaky_relu')

        # num_hidden_layers = 256
        # self.road_head = torch.nn.Sequential(
        #     nn.Conv2d(unet_out_ch,
        #               num_hidden_layers,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(num_hidden_layers, 1, kernel_size=3, stride=1,
        #               padding=1),
        #     nn.Sigmoid(),
        # )

        # Print input output layer dimensions
        self.example_input_array = torch.rand(
            (32, self.in_ch, self.in_size, self.in_size))

        if self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
        else:
            raise IOError(f'Undefined type ({type})')

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

    def binary_cross_entropy(self, x_hat, x, obs_mask=None, eps=1e-12):
        log_pxz = x * torch.log(x_hat +
                                eps) + (1. - x) * torch.log(1. - x_hat + eps)

        if obs_mask is not None:
            log_pxz = obs_mask * log_pxz

        # Avoid NaN for zero observation samples
        num_elems = (obs_mask.sum(dim=(-3, -2, -1)) + 1.)
        return log_pxz.sum(dim=(-3, -2, -1)) / num_elems

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

    @staticmethod
    def reorganize_viz_pairs(viz, num_pairs, res):
        '''
        Index mapping
            (0, 3) --> (0, 1)
            (1, 4) --> (2, 3)
            (2, 5) --> (4, 5)
        '''
        viz_reorg = np.zeros_like(viz)
        for idx in range(num_pairs):
            idx_0 = idx * res
            idx_1 = (idx + num_pairs) * res

            idx_2 = (2 * idx) * res
            idx_3 = (2 * idx + 1) * res
            viz_reorg[:, idx_2:idx_2 + res] = viz[:, idx_0:idx_0 + res]
            viz_reorg[:, idx_3:idx_3 + res] = viz[:, idx_1:idx_1 + res]
        viz = viz_reorg
        return viz

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
        h = self.encoder(x_in)
        x_hat = self.decoder(h)
        # 'road' and 'intensity' output head pair
        # road_pred = self.road_head(out_feat)

        return x_hat, h

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

        ###############
        #  Input x:s
        ###############
        x_in, x_oracle, x_target, m_target = self.unpack_sample(x)

        ######################
        #  Encoding vectors
        ######################
        h = self.encoder(x_in)
        x_hat = self.decoder(h)

        ################
        #  Prediction
        ################
        recon_loss = -self.binary_cross_entropy(x_hat, x_target, m_target)

        loss = recon_loss

        loss = loss.mean()  # + js.mean()

        # Variable metrics
        h_abs = torch.abs(h.detach())

        self.log_dict({
            'train_loss': loss,
            'train_recon_road': recon_loss.mean(),
            'train_h_abs': h_abs.mean(),
        })

        return loss

    def validation_step(self, batch, batch_idx):

        self.encoder.train()
        self.decoder.train()

        x, _ = batch

        x_in, x_oracle, x_target, m_target = self.unpack_sample(x)

        x_hats, h = self.forward(x_in)

        # Find smallest reconstruction loss among all modes
        recon_loss = -self.binary_cross_entropy(x_hats, x_target, m_target)

        self.log('val_recon', recon_loss.mean())

        self.logger.experiment.add_histogram('h', h, self.current_epoch)

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
            x_hats_ = []
            for _ in range(1):
                x_hats_.append(x_hats)
            x_hats = x_hats_

            # Avoid overshooting batch size
            view_num = 1
            num_viz_samples = min(x_in.shape[0],
                                  self.num_viz_samples) * view_num
            rows = 2

            x_in_viz = x_in.clone()
            x_in_viz[:, 0] = x_in_viz[:, 0] / 2 + 0.5
            viz = self.viz_mixture_preds(x_hats, x_in_viz, num_viz_samples)

            # Make columns be ordered by 'present / future' pairs
            res = x_in.shape[-2]
            num_pairs = num_viz_samples // 2
            viz = self.reorganize_viz_pairs(viz, num_pairs, res)

            size_per_fig = 4
            plt.figure(figsize=((num_viz_samples * size_per_fig,
                                 rows * size_per_fig)))
            plt.imshow(viz, vmin=0, vmax=1)
            plt.tight_layout()

            self.logger.experiment.add_figure('viz_det',
                                              plt.gcf(),
                                              global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        pass

    def viz_mixture_preds(self, x_hats: torch.Tensor, x: torch.Tensor,
                          num_viz_samples: int) -> np.array:
        '''
        Args:
            x_hats: List of K pred batch tensor (B,C,H,W) in interval (0,1)
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
        parser.add_argument('--in_ch', type=int, default=2)
        parser.add_argument('--in_size', type=int, default=128)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=256,
                            help='Encoder output dim')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--beta', type=float, default=1., help='For KL')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--encoder_conv_chs',
                            type=list,
                            default=[32, 64, 128, 256, 512, 128])
        parser.add_argument('--decoder_conv_chs',
                            type=list,
                            default=[64, 32, 16, 8, 4])
        parser.add_argument('--enc_str',
                            default='2x32,2x64,2x128,2x256,2x256,2x512,2x256')
        #                             128   64    32    16     8     4     4
        parser.add_argument('--dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32')
        #                                4     8    16    32   64  128
        parser.add_argument('--num_viz_samples', type=int, default=16)
        parser.add_argument('--num_log_p_samples', type=int, default=128)
        parser.add_argument('--test_sample_repeat', type=int, default=10)
        parser.add_argument('--sample_type', type=str, default='road')
        # parser.add_argument('', type=int, default=)
        return parent_parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    from datamodules.bev_datamodule import BEVDataModule

    parser = ArgumentParser()
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--val_data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    # Add program level args
    # Add model speficic args
    parser = DetPredModel.add_model_specific_args(parser)
    # Add all the vailable trainer option to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = DetPredModel(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)

    bev = BEVDataModule(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        do_rotation=True,
        do_extrapolation=False,
        do_masking=False,
        use_preproc=True,
        num_workers=args.num_workers,
    )

    trainer.fit(model, datamodule=bev)
