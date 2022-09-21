from typing import List

import torch
from torch import nn


class LargeMNISTExpEncoder(nn.Module):
    '''
    Ref: S3VAE encoder architecture
    '''

    def __init__(self,
                 in_ch,
                 enc_dim,
                 in_size,
                 conv_chs: List,
                 relu_slope: float = 0.2):
        super().__init__()

        self.in_ch = in_ch

        self.enc_dim = enc_dim
        self.in_size = in_size
        self.conv_chs = conv_chs
        self.relu_slope = relu_slope

        # Derived variables
        n_convs = len(self.conv_chs)
        # N-1 layers reduced by 1/2, last N layer reduces by 1/4
        self.feat_map_size = self.in_size // (2**(n_convs - 1) * 4)
        self.feat_map_flatten_dim = self.conv_chs[-1] * self.feat_map_size**2

        #############################
        #  Encoding transformations
        #############################
        layers = []
        conv_ch_prev = self.in_ch
        # Layers 1 --> N-1
        for conv_ch in self.conv_chs[:-1]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(conv_ch_prev,
                              out_channels=conv_ch,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(conv_ch),
                    nn.LeakyReLU(self.relu_slope),
                ))
            conv_ch_prev = conv_ch
        # Layer N
        conv_ch = self.conv_chs[-1]
        layers.append(
            nn.Sequential(
                nn.Conv2d(conv_ch_prev,
                          out_channels=conv_ch,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(conv_ch),
                nn.Tanh(),
            ))
        self.conv_encoder = nn.Sequential(*layers)

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.feat_map_flatten_dim, self.enc_dim, bias=False),
            nn.BatchNorm1d(self.enc_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # Encoding vector
        feat_map = self.conv_encoder(x)
        enc_vec = torch.flatten(feat_map, start_dim=1)
        enc_vec = self.fc_encoder(enc_vec)

        return enc_vec


if __name__ == '__main__':

    in_size = 128
    in_ch = 1
    enc_dim = 128
    conv_chs: List = [32, 64, 128, 256, 512, 128]
    relu_slope = 0.2

    encoder = LargeMNISTExpEncoder(in_ch, enc_dim, in_size, conv_chs,
                                   relu_slope)
    encoder.eval()

    x = torch.rand((1, in_ch, in_size, in_size))

    h = encoder(x)

    print(h.shape)
