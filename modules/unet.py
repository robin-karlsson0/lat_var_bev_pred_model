import pytorch_lightning as pl
import torch
from torch import nn


class UnetEncoder(pl.LightningModule):

    def __init__(
        self,
        enc_str: str,
        input_ch=1,
    ):
        '''
        Args:
            enc_str: Sequence of (#layers, #filters). Last pair is bottleneck.
                Ex: '2x64,2x128,2x256'
        '''
        super().__init__()

        enc_blocks = self.parse_enc_string(enc_str)

        ##############
        #  Encoders
        ##############
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch_prev = input_ch

        for num_layers, num_filters in enc_blocks[:-1]:

            enc_block = []
            for layer_idx in range(num_layers):
                enc_block.append(
                    nn.Conv2d(ch_prev,
                              num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False))
                enc_block.append(nn.BatchNorm2d(num_filters)),
                enc_block.append(nn.LeakyReLU())
                ch_prev = num_filters
            self.enc_blocks.append(nn.Sequential(*enc_block))

            downsampler = nn.Sequential(
                nn.Conv2d(ch_prev,
                          ch_prev,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(ch_prev),
                nn.LeakyReLU(),
            )
            self.downsamplers.append(downsampler)

        ################
        #  Bottleneck
        ################
        num_layers, num_filters = enc_blocks[-1]

        bottleneck = []
        for layer_idx in range(num_layers):
            bottleneck.append(
                nn.Conv2d(ch_prev,
                          num_filters,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            bottleneck.append(nn.BatchNorm2d(num_filters)),
            bottleneck.append(nn.LeakyReLU())
            ch_prev = num_filters
        self.bottleneck = nn.Sequential(*bottleneck)

        # <-- For latent variable predictive model
        self.fc_encoder = nn.Sequential(
            nn.Linear(4 * ch_prev, ch_prev, bias=False),
            nn.BatchNorm1d(ch_prev),
            nn.Tanh(),
        )
        # -->

    def forward(self, x):
        # Encoder
        encoder_outs = {}
        num_blocks = len(self.enc_blocks)
        for idx in range(num_blocks):
            encoder_out = self.enc_blocks[idx](x)
            x = self.downsamplers[idx](encoder_out)

            res = encoder_out.shape[-1]
            encoder_outs[res] = encoder_out

        # Bottleneck
        x = self.bottleneck(x)

        # return x, encoder_outs

        # <-- For latent variable predictive model
        x = x.flatten(start_dim=1)
        x = self.fc_encoder(x)
        return x
        # -->

    @staticmethod
    def parse_enc_string(enc_str):
        '''
        Args:
            enc_str: Sequence of (#layers, #filters)
                Ex: '2x64,2x128,2x256'
        '''
        enc_blocks = []
        s = enc_str.split(',')
        for ss in s:
            num_layers, num_filters = ss.split('x')
            enc_bloc = (int(num_layers), int(num_filters))
            enc_blocks.append(enc_bloc)

        return enc_blocks


class UnetDecoder(pl.LightningModule):

    def __init__(
        self,
        dec_str: str,
        bottleneck_ch,
        output_ch=1,
        input_ch=None,
        output_activation='sigmoid',
    ):
        '''
        Args:
            dec_str: Sequence of (#layers, #filters).
                Ex: '2x256,2x128,2x64'
            NOTE input_ch: Input vector dimension for latent variable
                           preditive model
        '''
        super().__init__()

        dec_blocks = self.parse_dec_string(dec_str)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=False)

        self.bottleneck_ch = bottleneck_ch

        ##############
        #  Decoders
        ##############
        self.dec_blocks = nn.ModuleList()
        ch_prev = self.bottleneck_ch

        for num_layers, num_filters in dec_blocks[:-1]:

            dec_block = []
            for layer_idx in range(num_layers):
                dec_block.append(
                    nn.Conv2d(ch_prev,
                              num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False))
                dec_block.append(nn.BatchNorm2d(num_filters)),
                dec_block.append(nn.LeakyReLU())
                ch_prev = num_filters
            self.dec_blocks.append(nn.Sequential(*dec_block))

        num_layers, num_filters = dec_blocks[-1]
        dec_block = []
        for layer_idx in range(num_layers - 1):
            dec_block.append(
                nn.Conv2d(ch_prev,
                          num_filters,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            dec_block.append(nn.BatchNorm2d(num_filters)),
            dec_block.append(nn.LeakyReLU())
            ch_prev = num_filters
        dec_block.append(
            nn.Conv2d(ch_prev, output_ch, kernel_size=3, stride=1, padding=1))
        if output_activation == 'sigmoid':
            dec_block.append(nn.Sigmoid())
        elif output_activation == 'leaky_relu':
            dec_block.append(nn.LeakyReLU())
        self.dec_blocks.append(nn.Sequential(*dec_block))

        # <-- Input decoder for latent variable predictive model
        feat_map_flatten_dim = self.bottleneck_ch * 2**2
        self.fc_decoder = nn.Sequential(
            nn.Linear(input_ch, input_ch, bias=False),
            nn.BatchNorm1d(input_ch),
            nn.LeakyReLU(),
            nn.Linear(input_ch, feat_map_flatten_dim, bias=False),
            nn.BatchNorm1d(feat_map_flatten_dim),
            nn.LeakyReLU(),
        )
        # -->

    def forward(self, x):
        '''
        TODO:  Add in optional 'enc_outs: dict'
        '''

        # <-- For latent variable predictive model
        x = self.fc_decoder(x)
        x = x.reshape(-1, self.bottleneck_ch, 2, 2)
        # -->

        num_blocks = len(self.dec_blocks)
        for idx in range(num_blocks):
            x = self.upsample(x)

            # res = x.shape[-1]
            # enc_out = enc_outs[res]
            # x = torch.cat((x, enc_out), dim=1)
            x = self.dec_blocks[idx](x)

        return x

    @staticmethod
    def parse_dec_string(dec_str):
        '''
        Args:
            enc_str: Sequence of (#layers, #filters)
                Ex: '2x64,2x128,2x256'
        '''
        dec_blocks = []
        s = dec_str.split(',')
        for ss in s:
            num_layers, num_filters = ss.split('x')
            enc_bloc = (int(num_layers), int(num_filters))
            dec_blocks.append(enc_bloc)

        return dec_blocks


# class Unet(pl.LightningModule):
#     '''
#     '''
#
#     def __init__(self,
#                  input_size: int,
#                  base_ch=64,
#                  input_ch=1,
#                  dropout_prob=0.):
#         super().__init__()
#
#         self.encoder = UnetEncoder(input_size, base_ch, input_ch, dropout_prob)
#         self.decoder = UnetDecoder(
#             input_size,
#             base_ch,
#             input_ch,
#             dropout_prob,
#         )
#
#     def forward(self, x):
#         pass

if __name__ == '__main__':

    input_size = 128
    base_ch = 2
    input_ch = 1

    #############
    #  Encoder
    #############
    enc_str = '2x32,2x64,2x128,2x256,2x256,2x512,2x512'
    #           128   64    32    16     8     4     4
    unet_encoder = UnetEncoder(enc_str)

    x = torch.rand((32, input_ch, input_size, input_size))
    x_bottleneck, enc_outs = unet_encoder(x)

    print(f'{x.shape} --> {x_bottleneck.shape}')
    for res in enc_outs.keys():
        print(f'enc_out[{res}] : {enc_outs[res].shape}')

    #############
    #  Decoder
    #############
    dec_str = '2x512,2x256,2x256,2x128,2x64,2x32'
    #              4     8    16    32   64  128
    bottleneck_ch = 512
    output_ch = 1
    unet_decoder = UnetDecoder(dec_str, bottleneck_ch, output_ch)

    y = unet_decoder(x_bottleneck)

    print(f'{x_bottleneck.shape} --> {y.shape}')
