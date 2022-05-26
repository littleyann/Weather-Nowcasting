import torch
from torch import nn
from ConvGRU import ConvGRU
from Blocks import GBlock
from einops.layers.torch import Rearrange


class Decoder(nn.Module):
    def __init__(self, h_state_dims, x_dims):
        super(Decoder, self).__init__()

        self.gru1 = ConvGRU(h_state_dim=h_state_dims[0], x_dim=x_dims[0], kernel_size=(3, 3))
        self.gru2 = ConvGRU(h_state_dim=h_state_dims[1], x_dim=x_dims[1], kernel_size=(3, 3))
        self.gru3 = ConvGRU(h_state_dim=h_state_dims[2], x_dim=x_dims[2], kernel_size=(3, 3))
        self.gru4 = ConvGRU(h_state_dim=h_state_dims[3], x_dim=x_dims[3], kernel_size=(3, 3))

        self.g_block1 = nn.Sequential(
            nn.Conv2d(h_state_dims[0], x_dims[0], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(x_dims[0], x_dims[0], False),
            GBlock(x_dims[0], x_dims[1], True),
        )

        self.g_block2 = nn.Sequential(
            nn.Conv2d(h_state_dims[1], x_dims[1], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(x_dims[1], x_dims[1], False),
            GBlock(x_dims[1], x_dims[2], True),
        )

        self.g_block3 = nn.Sequential(
            nn.Conv2d(h_state_dims[2], x_dims[2], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(x_dims[2], x_dims[2], False),
            GBlock(x_dims[2], x_dims[3], True),
        )

        self.g_block4 = nn.Sequential(
            nn.Conv2d(h_state_dims[3], x_dims[3], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(x_dims[3], x_dims[3], False),
            GBlock(x_dims[3], x_dims[3], True),
        )

        self.out = nn.Sequential(
            nn.BatchNorm2d(x_dims[3]),
            nn.ReLU(),
            nn.Conv2d(x_dims[3], 4, kernel_size=(1, 1)),
            nn.modules.PixelShuffle(upscale_factor=2),
        )

    def forward(self, x, hidden_stats):
        gru1_outs = self.gru1(x, hidden_stats[3])
        gru1_outs = [self.g_block1(gru1_out) for gru1_out in gru1_outs]
        gru1_out = torch.stack(gru1_outs, dim=1)

        gru2_outs = self.gru2(gru1_out, hidden_stats[2])
        gru2_outs = [self.g_block2(gru2_out) for gru2_out in gru2_outs]
        gru2_out = torch.stack(gru2_outs, dim=1)

        gru3_outs = self.gru3(gru2_out, hidden_stats[1])
        gru3_outs = [self.g_block3(gru3_out) for gru3_out in gru3_outs]
        gru3_out = torch.stack(gru3_outs, dim=1)

        gru4_outs = self.gru4(gru3_out, hidden_stats[0])
        gru4_outs = [self.g_block4(gru4_out) for gru4_out in gru4_outs]

        outs = [self.out(gru4_out) for gru4_out in gru4_outs]
        out = torch.stack(outs, dim=1)

        return out


if __name__ == '__main__':
    input_tensor = torch.randn((2, 20, 512, 15, 18))
    hidden_tensors = [torch.randn((2, 32, 120, 144)), torch.randn((2, 64, 60, 72)), torch.randn((2, 128, 30, 36)),
                      torch.randn((2, 256, 15, 18))]

    model = Decoder(
        h_state_dims=[256, 128, 64, 32],
        x_dims=[512, 256, 128, 64]
    )

    output_tensor = model(input_tensor, hidden_tensors)
    print('last gru output is: {}'.format(output_tensor.shape))



