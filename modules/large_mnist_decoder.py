from typing import List

import torch
from torch import nn


class LargeMNISTExpDecoder(nn.Module):
    '''
    '''

    def __init__(self, out_ch, enc_dim, lat_dim, in_size, conv_chs: List):
        super().__init__()

        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.lat_dim = lat_dim
        self.in_size = in_size
        self.conv_chs = conv_chs

        # Derived variables
        n_convs = len(self.conv_chs)
        # Like encoder but increased by 2x to account for 2x larger feature dim
        encoder_map_size = self.in_size // (2**(n_convs - 2) * 4)
        self.feat_map_size = encoder_map_size * 2
        self.feat_map_flatten_dim = self.conv_chs[0] * self.feat_map_size**2

        ##############################
        #  Decoding transformations
        ##############################

        self.fc_decoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.enc_dim, bias=False),
            nn.BatchNorm1d(self.enc_dim),
            nn.LeakyReLU(),
            nn.Linear(self.enc_dim, self.feat_map_flatten_dim, bias=False),
            nn.BatchNorm1d(self.feat_map_flatten_dim),
            nn.LeakyReLU(),
        )

        self.ch = 4
        self.relu = nn.LeakyReLU()

        layers = []
        # Layer 1
        conv_ch_prev = self.conv_chs[0]
        conv_ch = self.conv_chs[1]
        layers.append(
            nn.Sequential(
                nn.Conv2d(conv_ch_prev,
                          out_channels=conv_ch,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(conv_ch),
                nn.ReLU(),
                nn.Conv2d(conv_ch,
                          out_channels=conv_ch,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(conv_ch),
                nn.ReLU(),
            ))
        # Layers 2 --> N-1
        conv_ch_prev = conv_ch
        for conv_ch in self.conv_chs[1:-1]:
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2,
                                mode="bilinear",
                                align_corners=False),
                    nn.Conv2d(conv_ch_prev,
                              out_channels=conv_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(conv_ch),
                    nn.ReLU(),
                    nn.Conv2d(conv_ch,
                              out_channels=conv_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(conv_ch),
                    nn.ReLU(),
                ))
            conv_ch_prev = conv_ch
        # Layer N
        conv_ch_prev = self.conv_chs[-2]
        conv_ch = self.conv_chs[-1]
        layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode="bilinear",
                            align_corners=False),
                nn.Conv2d(conv_ch_prev,
                          out_channels=conv_ch,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(conv_ch),
                nn.ReLU(),
                nn.Conv2d(conv_ch,
                          out_channels=conv_ch,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(conv_ch),
                nn.ReLU(),
                nn.Conv2d(conv_ch,
                          out_channels=self.out_ch,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.Sigmoid(),
            ))
        self.conv_decoder = nn.Sequential(*layers)

    def forward(self, lat_vec):
        # Encoding vector
        enc_vec = self.fc_decoder(lat_vec)
        # Output map
        feat_map = enc_vec.reshape(-1, self.conv_chs[0], self.feat_map_size,
                                   self.feat_map_size)
        out = self.conv_decoder(feat_map)

        return out


if __name__ == '__main__':

    out_ch = 1
    enc_dim = 128
    lat_dim = 8
    in_size = 128
    conv_chs = [512, 256, 128, 64, 32]

    decoder = LargeMNISTExpDecoder(out_ch, enc_dim, lat_dim, in_size, conv_chs)
    decoder.eval()

    h = torch.rand((1, 8))

    y = decoder(h)

    print(y.shape)
