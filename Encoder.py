import torch
from torch import nn
from Blocks import DBlock, GBlock
import einops


class Encoder(nn.Module):
    def __init__(self, in_chan, chan, time_steps):
        super(Encoder, self).__init__()

        self.space2depth = nn.modules.PixelUnshuffle(downscale_factor=2)

        self.d_block1 = DBlock(4*in_chan, chan)
        self.d_block2 = DBlock(chan, chan*2)
        self.d_block3 = DBlock(chan*2, chan*4)
        self.d_block4 = DBlock(chan*4, chan*8)

        self.conv1 = nn.Conv2d(chan*time_steps, chan, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(chan*2*time_steps, chan*2, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(chan*4*time_steps, chan*4, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(chan*8*time_steps, chan*8, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.space2depth(x)
        time_steps = x.size(1)
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []

        for i in range(time_steps):
            s1 = self.d_block1(x[:, i, :, :, :])
            # print('s1 shape: {}'.format(s1.shape))
            s2 = self.d_block2(s1)
            s3 = self.d_block3(s2)
            s4 = self.d_block4(s3)
            # print('s4 shape: {}'.format(s4.shape))

            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)

        scale_1 = torch.stack(scale_1, dim=1)
        # print('scale1 shape: {}'.format(scale_1.shape))
        scale_2 = torch.stack(scale_2, dim=1)
        scale_3 = torch.stack(scale_3, dim=1)
        scale_4 = torch.stack(scale_4, dim=1)
        # print('scale4 shape: {}'.format(scale_4.shape))

        scale_1 = self.conv1(Encoder.reshape(scale_1))
        scale_2 = self.conv2(Encoder.reshape(scale_2))
        scale_3 = self.conv3(Encoder.reshape(scale_3))
        scale_4 = self.conv4(Encoder.reshape(scale_4))
        return [scale_1, scale_2, scale_3, scale_4]

    @staticmethod
    def reshape(inputs):
        output = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return output


if __name__ == '__main__':
    input_tensor = torch.randn((2, 20, 1, 480, 576))
    encoder = Encoder(1, 32, 20)
    output_tensors = encoder(input_tensor)
    print(output_tensors[0].shape)
    print(output_tensors[1].shape)
    print(output_tensors[2].shape)
    print(output_tensors[3].shape)
