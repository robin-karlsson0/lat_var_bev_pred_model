import os
import pickle
import random

import matplotlib as mpl
# mpl.use('agg')  # Must be before pyplot import to avoid memory leak
import matplotlib.pyplot as plt
import numpy as np
# import onnx
# import onnxruntime as ort
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from det_var_pred_model_prod import DetPredModelProd
from lat_var_pred_model_prod import LatVarPredModelProd
from modules.unet_gan import UnetDecoder, UnetEncoder
from utils.lat_var_pred_aux import integrate_obs_and_lat_pred


class RoadAdvPredModel(pl.LightningModule):

    def __init__(
        self,
        batch_size,
        in_ch,
        in_size,
        out_ch,
        enc_dim,
        lr,
        weight_decay,
        gen_enc_str,
        gen_dec_str,
        adv_enc_str,
        adv_dec_str,
        num_viz_samples,
        num_log_p_samples,
        sample_type,
        objective_type,
        onnx_filename,
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
        self.weight_decay = weight_decay
        self.num_viz_samples = num_viz_samples

        self.num_log_p_samples = num_log_p_samples
        self.prec_rec_samples = 10000
        self.unknown_viz_val = 0.5
        self.sample_type = sample_type

        if objective_type == 'bce':
            self.objective = self.binary_cross_entropy
        elif objective_type == 'mse':
            self.objective = self.mse
        else:
            raise IOError(f'Undefined objective type ({objective_type})')

        ############
        #  Models
        ############
        # Latent variable predictive model
        # if os.path.isfile(onnx_filename) is not True:
        #     raise IOError(f'ONNX model does not exist ({onnx_filename})')
        # self.ort_lat_var_pred_model = ort.InferenceSession(
        #     onnx_filename,
        #     providers=['CPUExecutionProvider'])  # CUDAExecutionProvider
        # with open('det_var_pred_model_config.pkl', 'rb') as f:
        #     model_config_dict = pickle.load(f)
        with open('lat_var_pred_model_config.pkl', 'rb') as f:
            model_config_dict = pickle.load(f)
        # self.road_pred_model = DetPredModelProd(**model_config_dict)
        # checkpoint = torch.load('epoch=999-step=701000.ckpt')
        self.road_pred_model = LatVarPredModelProd(**model_config_dict)
        checkpoint = torch.load('models/version_5_ep_999.ckpt')
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

        # Adversarial Unet
        self.adv_encoder = UnetEncoder(adv_enc_str, input_ch=in_ch)
        vec_dim = int(adv_dec_str.split(',')[0].split('x')[-1])
        unet_out_ch = 1  # int(dec_str.split(',')[-1].split('x')[-1])
        self.adv_decoder = UnetDecoder(adv_enc_str,
                                       adv_dec_str,
                                       vec_dim,
                                       output_ch=unet_out_ch,
                                       output_activation='sigmoid')

        # Flattened feature map vector dimension
        # bottleneck_dim = self.enc_dim * 4
        # self.fc_bottleneck = torch.nn.Sequential(
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     nn.LeakyReLU(),
        # )

        # Print input output layer dimensions
        self.example_input_array = torch.rand(
            (32, self.in_ch, self.in_size, self.in_size))

        if self.sample_type == 'all':
            self.unpack_sample = self.unpack_sample_all
        elif self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
        else:
            raise IOError(f'Undefined type ({type})')

    def configure_optimizers(self):
        model_params = list(self.encoder.parameters()) + list(
            self.decoder.parameters())
        opt_model = torch.optim.Adam(model_params,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999))

        discr_params = list(self.adv_encoder.parameters()) + list(
            self.adv_decoder.parameters())
        opt_critic = torch.optim.Adam(discr_params,
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=(0.5, 0.999))
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=self.lr,
        #                             weight_decay=self.weight_decay)
        return (
            {
                'optimizer': opt_model,
                'frequency': 1
            },
            {
                'optimizer': opt_critic,
                'frequency': 1
            },
        )

    def binary_cross_entropy(self, x_hat, x, obs_mask=None, eps=1e-12):
        log_pxz = x * torch.log(x_hat +
                                eps) + (1. - x) * torch.log(1. - x_hat + eps)

        if obs_mask is not None:
            log_pxz = obs_mask * log_pxz

        # Avoid NaN for zero observation samples
        num_elems = (obs_mask.sum(dim=(-3, -2, -1)) + 1.)
        return -log_pxz.sum(dim=(-3, -2, -1)) / num_elems

    def mse(self, x_hat, x, obs_mask=None):
        mse = (x - x_hat)**2

        if obs_mask is not None:
            mse = obs_mask * mse

        # Avoid NaN for zero observation samples
        num_elems = (obs_mask.sum(dim=(-3, -2, -1)) + 1.)
        return mse.sum(dim=(-3, -2, -1)) / num_elems

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

    def forward(self, x):
        '''
        Model:
            x --> enc() --> h --> fc() --> h' --> dec() --> y

        Args:
            x: (B, 2, H, W)
                [0]: Road
                [1]: Intensity
        '''
        h, enc_outs = self.encoder(x)

        # feat_map_shape = h.shape
        # h = h.view(B, -1)
        # h = self.fc_bottleneck(h)
        # h = h.view(feat_map_shape)

        y = self.decoder(h, enc_outs)
        return y

    def forward_adv(self, x):
        '''
        Model:
            x --> enc() --> h --> fc() --> h' --> dec() --> y

        Args:
            x: (B, 2, H, W)
                [0]: Road
                [1]: Intensity
        '''
        h, enc_outs = self.adv_encoder(x)

        # feat_map_shape = h.shape
        # h = h.view(B, -1)
        # h = self.fc_bottleneck(h)
        # h = h.view(feat_map_shape)

        y = self.adv_decoder(h, enc_outs)
        return y

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        # device = x.get_device()

        #########################
        #  Representations
        #    1. Obs. road
        #    2. Obs. intensity
        #########################
        x_in, x_oracle, x_target, m_target = self.unpack_sample(x)
        x_oracle_road, x_oracle_int = x_oracle.chunk(2, dim=1)
        m_oracle_road, m_oracle_int = m_target.chunk(2, dim=1)

        # For adding intensity pred
        x_oracle_int_adv = x_oracle_int.clone()

        ##########################
        #  Structure prediction
        #    1. Pred. road
        ##########################
        # ort_input = {'input': x_oracle_road.cpu().numpy()}
        # x_pred_road = self.ort_lat_var_pred_model.run(None, ort_input)
        # x_pred_road = x_pred_road[0]  # Extract from list
        # x_pred_road = torch.tensor(x_pred_road, device=device)
        with torch.no_grad():
            x_pred_road = self.road_pred_model.forward(x_oracle_road)[0]
            mode_idx = 0  # random.randint(0, self.road_pred_model.y_dim - 1)
            x_pred_road = x_pred_road[:, mode_idx]

        ########################
        #  Texture prediction
        ########################
        # Replace 'observed' road with 'obs + pred' road
        x_pred_road = integrate_obs_and_lat_pred(x_oracle_road, x_pred_road,
                                                 m_oracle_road)

        x_oracle_aug = x_oracle.clone()
        x_oracle_aug[:, 0:1] = x_pred_road

        m_fake = torch.logical_xor(x_pred_road, m_oracle_int)

        # Train generator
        if optimizer_idx == 0:
            x_pred_int = self.forward(x_oracle_aug)

            # recon_loss = self.mse(x_pred_int, x_oracle_int, m_oracle_int)

            x_oracle_int_adv[m_fake] = x_pred_int[m_fake]
            x_oracle_int_adv = torch.concat([x_pred_road, x_oracle_int_adv],
                                            dim=1)

            x_oracle_int_adv = 2 * x_oracle_int_adv - 1
            real_pred = self.forward_adv(x_oracle_int_adv)

            # Want the generator to fool discriminator so that all fake
            # elements (0) are predicted as real elements (1)
            #     ==> Optimize {fake elements} --> 1
            gen_loss = self.binary_cross_entropy(real_pred,
                                                 torch.ones_like(real_pred),
                                                 m_fake)

            # loss = recon_loss + gen_loss
            loss = gen_loss
            loss = loss.mean()

            self.log_dict({
                # 'train_recon': recon_loss.mean(),
                'train_gen': gen_loss.mean(),
            })

            return loss

        # Train discriminator
        if optimizer_idx == 1:

            with torch.no_grad():
                x_pred_int = self.forward(x_oracle_aug)

            x_oracle_int_adv[m_fake] = x_pred_int[m_fake]
            x_oracle_int_adv = torch.concat([x_pred_road, x_oracle_int_adv],
                                            dim=1)

            x_oracle_int_adv = 2 * x_oracle_int_adv - 1
            real_pred = self.forward_adv(x_oracle_int_adv)

            # Want discriminator to correctly predict real (1) and fake (0)
            # elements
            loss_adv = self.binary_cross_entropy(real_pred,
                                                 m_oracle_int.float(),
                                                 x_pred_road)
            loss_adv = loss_adv.mean()

            m_real = m_oracle_int.bool()
            real_pred_acc, fake_pred_acc = self.comp_adv_acc(
                real_pred, m_real, m_fake)

            self.log_dict({
                'train_adv': loss_adv,
                'val_real_acc': real_pred_acc.mean(),
                'val_fake_acc': fake_pred_acc.mean(),
            })

            return loss_adv

    def validation_step(self, batch, batch_idx):

        x, _ = batch
        # device = x.get_device()

        #########################
        #  Representations
        #    1. Obs. road
        #    2. Obs. intensity
        #########################
        x_in, x_oracle, x_target, m_target = self.unpack_sample(x)
        x_oracle_road, x_oracle_int = x_oracle.chunk(2, dim=1)
        m_oracle_road, m_oracle_int = m_target.chunk(2, dim=1)

        ##########################
        #  Structure prediction
        #    1. Pred. road
        ##########################
        # ort_input = {'input': x_oracle_road.cpu().numpy()}
        # x_pred_road = self.ort_lat_var_pred_model.run(None, ort_input)
        # x_pred_road = x_pred_road[0]  # Extract from list
        # x_pred_road = torch.tensor(x_pred_road, device=device)
        with torch.no_grad():
            x_pred_road = self.road_pred_model.forward(x_oracle_road)[0]
            mode_idx = 0  # random.randint(0, self.road_pred_model.y_dim - 1)
            x_pred_road = x_pred_road[:, mode_idx]

        ########################
        #  Texture prediction
        ########################
        # Replace 'observed' road with 'obs + pred' road
        x_pred_road = integrate_obs_and_lat_pred(x_oracle_road, x_pred_road,
                                                 m_oracle_road)
        x_oracle[:, 0:1] = x_pred_road

        m_real = m_oracle_int.bool()
        m_fake = torch.logical_xor(x_pred_road, m_oracle_int)
        m_all = torch.logical_or(m_real, m_fake)

        x_pred_int = self.forward(x_oracle)

        x_oracle_int[m_fake] = x_pred_int[m_fake]
        x_oracle = torch.concat([x_pred_road, x_oracle_int], dim=1)

        x_oracle_norm = 2 * x_oracle - 1
        real_pred = self.forward_adv(x_oracle_norm)  # (B, 1, H, W)

        # recon_loss = self.mse(x_pred_int, x_oracle_int, m_oracle_int)
        adv_loss = self.binary_cross_entropy(real_pred, m_oracle_int.float(),
                                             x_pred_road)

        real_pred_acc, fake_pred_acc = self.comp_adv_acc(
            real_pred, m_real, m_fake)

        # loss = recon_loss  # + adv_loss
        # loss = adv_loss
        # loss = loss.mean()

        # self.log('val_loss', loss, sync_dist=True)
        # self.log('val_recon', recon_loss.mean(), sync_dist=True)
        self.log('val_adv', adv_loss.mean(), sync_dist=True)
        self.log('val_real_acc', real_pred_acc.mean(), sync_dist=True)
        self.log('val_fake_acc', fake_pred_acc.mean(), sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 1 == 0:
            #####################################
            #  Visualize input and predictions
            #####################################
            x_pred_ints = []
            for _ in range(1):
                x_pred_ints.append(x_oracle_int)
                real_pred[~m_all] = 0.5  # * torch.ones_like(real_pred)[~m_all]
                x_pred_ints.append(real_pred)

            # Avoid overshooting batch size
            view_num = 1
            num_viz_samples = min(x_oracle_int.shape[0],
                                  self.num_viz_samples) * view_num

            x_in_viz = x_oracle_road.clone()
            x_in_viz = 0.5 * (x_in_viz + 1) - 0.5
            x_in_viz[m_oracle_int] = x_oracle[:, 1:2][m_oracle_int]
            viz = self.viz_mixture_preds(x_pred_ints, x_in_viz,
                                         num_viz_samples)

            # Make columns be ordered by 'present / future' pairs
            res = x_in.shape[-2]
            num_pairs = num_viz_samples // 2
            viz = self.reorganize_viz_pairs(viz, num_pairs, res)

            rows = 3
            size_per_fig = 8
            plt.figure(figsize=((num_viz_samples * size_per_fig,
                                 rows * size_per_fig)))
            plt.imshow(viz, vmin=0, vmax=1)
            plt.tight_layout()

            # for sample_idx in range(num_viz_samples):
            #     for k in range(10):
            #         msg = str(k)
            #         h_pos = sample_idx * self.in_size * 2
            #         v_pos = (k + 1) * self.in_size + 3
            #         plt.text(h_pos, v_pos, msg, color='white')

            self.logger.experiment.add_figure('viz',
                                              plt.gcf(),
                                              global_step=self.current_epoch)

    @staticmethod
    def comp_adv_acc(real_pred,
                     m_real,
                     m_fake,
                     real_thresh=0.75,
                     fake_thresh=0.25):
        '''
        Args:
            real_pred: Predicted real and fake element prob map (B,1,H,W)
            m_real: Real element GT map (B,1,H,W)
            m_fake: Fake element GT map (B,1,H,W)
        Returns:
            Accuracy scores (real, fake)
        '''
        # Only count confident predictions
        real_pred_true = real_pred > real_thresh
        real_pred_false = real_pred < fake_thresh

        # Get correctly predicted elements using GT masks
        real_pred_true = torch.logical_and(real_pred_true, m_real)
        real_pred_false = torch.logical_and(real_pred_false, m_fake)

        # Compute accuracy score
        real_pred_true = real_pred_true.sum(dim=(1, 2, 3)) / (
            (m_real).sum(dim=(1, 2, 3)) + 1)
        real_pred_false = real_pred_false.sum(dim=(1, 2, 3)) / (
            (m_fake).sum(dim=(1, 2, 3)) + 1)

        return real_pred_true, real_pred_false

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
        parser = parent_parser.add_argument_group('RoadAdvPredModel')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--in_ch', type=int, default=2)
        parser.add_argument('--in_size', type=int, default=128)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=256,
                            help='Encoder output dim')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--gen_enc_str',
                            default='2x32,2x64,2x128,2x256,2x256,2x512,2x256')
        #                             128   64    32    16     8     4     4
        parser.add_argument('--gen_dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32')
        #                                4     8    16    32   64  128
        parser.add_argument('--adv_enc_str',
                            default='2x32,2x64,2x128,2x256,2x256,2x512,2x256')
        #                             128   64    32    16     8     4     4
        parser.add_argument('--adv_dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32')
        #                                4     8    16    32   64  128
        parser.add_argument('--num_viz_samples', type=int, default=16)
        parser.add_argument('--num_log_p_samples', type=int, default=128)
        parser.add_argument('--test_sample_repeat', type=int, default=10)
        parser.add_argument('--sample_type', type=str, default='road')
        # parser.add_argument('', type=int, default=)
        parser.add_argument('--objective_type', type=str, default='mse')
        parser.add_argument('--onnx_filename', type=str)
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
    parser = RoadAdvPredModel.add_model_specific_args(parser)
    # Add all the vailable trainer option to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = RoadAdvPredModel(**dict_args)
    # model = RoadAdvPredModel.load_from_checkpoint(
    #     'lightning_logs/version_0/checkpoints/epoch=14-step=4695.ckpt')
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