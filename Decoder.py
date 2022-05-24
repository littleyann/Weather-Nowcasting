import torch
from torch import nn
from ConvGRU import ConvGRU
from Blocks import GBlock
from einops.layers.torch import Rearrange


class Decoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(Decoder, self).__init__()

        self.gru1 = ConvGRU(input_dim=dim[0], hidden_dim=hidden_dim[0], kernel_size=(3, 3))
        self.gru2 = ConvGRU(input_dim=dim[1], hidden_dim=hidden_dim[1], kernel_size=(3, 3))
        self.gru3 = ConvGRU(input_dim=dim[2], hidden_dim=hidden_dim[2], kernel_size=(3, 3))
        self.gru4 = ConvGRU(input_dim=dim[3], hidden_dim=hidden_dim[3], kernel_size=(3, 3))

        self.g_block1 = nn.Sequential(
            nn.Conv2d(hidden_dim[0], dim[0], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(dim[0], dim[0], False),
            GBlock(dim[0], dim[1], True),
        )

        self.g_block2 = nn.Sequential(
            nn.Conv2d(hidden_dim[1], dim[1], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(dim[1], dim[1], False),
            GBlock(dim[1], dim[2], True),
        )

        self.g_block3 = nn.Sequential(
            nn.Conv2d(hidden_dim[2], dim[2], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(dim[2], dim[2], False),
            GBlock(dim[2], dim[3], True),
        )

        self.g_block4 = nn.Sequential(
            nn.Conv2d(hidden_dim[3], dim[3], kernel_size=(1, 1), padding=(0, 0)),
            GBlock(dim[3], dim[3], False),
            GBlock(dim[3], dim[3], True),
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
        gru4_out = torch.stack(gru4_outs, dim=1)

        return gru4_out


if __name__ == '__main__':
    input_tensor = torch.randn((2, 20, 512, 15, 18))
    hidden_tensors = [torch.randn((2, 32, 120, 144)), torch.randn((2, 64, 60, 72)), torch.randn((2, 128, 30, 36)),
                      torch.randn((2, 256, 15, 18))]

    model = Decoder(
        dim=[512, 256, 128, 64],
        hidden_dim=[256, 128, 64, 32]
    )

    output_tensor = model(input_tensor, hidden_tensors)
    print('last gru output is: {}'.format(output_tensor.shape))



