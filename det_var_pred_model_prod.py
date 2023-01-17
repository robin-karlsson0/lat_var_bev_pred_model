# import matplotlib as mpl

# mpl.use('agg')  # Must be before pyplot import to avoid memory leak
# import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import torch
import torch.nn as nn

from modules.unet import UnetDecoder, UnetEncoder
from utils.lat_var_pred_aux import integrate_obs_and_lat_pred


class DetPredModelProd(nn.Module):
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
        enc_str,
        dec_str,
        sample_type,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.in_ch = in_ch
        self.in_size = in_size
        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.sample_type = sample_type

        ############
        #  Models
        ############
        # Unet
        self.encoder = UnetEncoder(enc_str, input_ch=self.in_ch)
        vec_dim = self.enc_dim
        unet_out_ch = 1  # int(dec_str.split(',')[-1].split('x')[-1])
        self.decoder = UnetDecoder(dec_str,
                                   vec_dim,
                                   input_ch=vec_dim,
                                   output_ch=unet_out_ch,
                                   output_activation='sigmoid')

        # Print input output layer dimensions
        self.example_input_array = torch.rand(
            (32, self.in_ch, self.in_size, self.in_size))

        if self.sample_type == 'road':
            self.unpack_sample = self.unpack_sample_road
        else:
            raise IOError(f'Undefined type ({type})')

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DetPredModel')
        parser.add_argument('--in_ch', type=int, default=2)
        parser.add_argument('--in_size', type=int, default=128)
        parser.add_argument('--out_ch', type=int, default=1)
        parser.add_argument('--enc_dim',
                            type=int,
                            default=256,
                            help='Encoder output dim')
        parser.add_argument('--enc_str',
                            default='2x32,2x64,2x128,2x256,2x256,2x512,2x256')
        #                             128   64    32    16     8     4     4
        parser.add_argument('--dec_str',
                            default='2x512,2x256,2x256,2x128,2x64,2x32')
        #                                4     8    16    32   64  128
        parser.add_argument('--sample_type', type=str, default='road')
        # parser.add_argument('', type=int, default=)
        return parent_parser


if __name__ == '__main__':
    import os
    import pickle
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    import onnx
    import onnxruntime as ort

    from datamodules.bev_datamodule import BEVDataModule

    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--make_onnx_model', action="store_true")
    parser.add_argument('--use_oracle_sample', action="store_true")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    # Add model speficic args
    parser = DetPredModelProd.add_model_specific_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    model = DetPredModelProd(**dict_args)
    with open('det_var_pred_model_config.pkl', 'wb') as f:
        pickle.dump(dict_args, f)

    exit()

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # model.cuda()
    model.eval()

    #######################
    #  Create ONNX model
    #######################
    onnx_filename = f'det_pred_model_bs{args.batch_size}.onnx'
    if args.make_onnx_model:
        x = torch.randn(args.batch_size, 1, 256, 256, requires_grad=True)
        torch.onnx.export(
            model,
            x,
            onnx_filename,
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

    if os.path.isfile(onnx_filename) is not True:
        raise IOError(f'ONNX model does not exist ({onnx_filename})')

    # Check that the model is well formed
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

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
    ort_session = ort.InferenceSession(onnx_filename,
                                       providers=['CUDAExecutionProvider'])

    for idx, batch in enumerate(dataloader):

        print(f'idx {idx}')

        x, _ = batch

        x_in, x_oracle, x_target, m_target = model.unpack_sample_road(x)
        if args.use_oracle_sample:
            x_in = x_oracle
        # Remove future sample
        x_in = x_in[0:args.batch_size]
        x_target = x_target[0:args.batch_size]
        m_target = m_target[0:args.batch_size]
        # x_in = x_in.cuda()
        # x_oracle = x_oracle[0:1]
        # x_oracle = x_oracle.cuda()

        outputs = ort_session.run(None, {'input': x_in.numpy()})
        x_hat = outputs[0]  # (B, 1, H, W)

        x_in = x_in.cpu().numpy()
        x_target = x_target.cpu().numpy()
        m_target = m_target.cpu().numpy()
        # x_oracle = x_oracle.cpu().numpy()
        # x_hats = [x_hat.detach().cpu().numpy() for x_hat in x_hats]

        # x_hat1 = torch.tensor(x_hat)
        x_hat1 = integrate_obs_and_lat_pred(x_target,
                                            x_hat,
                                            m_target,
                                            is_np=True)
        # x_hat1 = x_hat.copy()
        # x_hat1[m_target] = x_target[m_target]

        # mask = x_hat1 < 0.25
        # x_hat1[mask] = 0
        # mask = x_hat1 > 0.25
        # x_hat1[mask] = 1

        x_in_real, _, _, _ = model.unpack_sample_road(x)

        num_rows = 4
        num_cols = args.batch_size
        B = x_hat.shape[0]
        for idx in range(B):
            plt.subplot(num_rows, B, idx + 1 + 0 * B)
            plt.imshow(x_in_real[idx, 0])

            plt.subplot(num_rows, B, idx + 1 + 1 * B)
            plt.imshow(x_in[idx, 0])

            plt.subplot(num_rows, B, idx + 1 + 2 * B)
            plt.imshow(x_hat[idx, 0])

            plt.subplot(num_rows, B, idx + 1 + 3 * B)
            plt.imshow(x_hat1[idx, 0])

        plt.tight_layout()
        plt.show()
