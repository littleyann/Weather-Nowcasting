import torch
from torch import nn
from Encoder import Encoder
from Decoder import Decoder
from GenerateInput import Input
import einops


class Model(nn.Module):
    def __init__(self, in_chan, encoder_chan_rise, time_steps, generate_channels, num_l_blocks, input_shape,
                 h_state_dims, x_dims):
        """
        :param in_chan: input_channel number of Model
        :param encoder_chan_rise: the step length of each channel risen
        :param time_steps: total time steps
        :param generate_channels: channels for l_blocks when generate input
        :param num_l_blocks: the number of l_block
        :param input_shape: the initial input shape
        :param h_state_dims: the output dims of encoder at each stage
        :param x_dims: the hidden state dim at each stage
        """

        super(Model, self).__init__()

        self.encoder = Encoder(in_chan, encoder_chan_rise, time_steps)
        self.generate_input = Input(generate_channels, num_l_blocks, input_shape)
        self.decoder = Decoder(h_state_dims, x_dims)

    def forward(self, x):
        hidden_states = self.encoder(x)

        inputs = self.generate_input(x)
        inputs = einops.repeat(inputs, "b c h w -> (repeat b) c h w", repeat=hidden_states[0].shape[0])
        inputs = inputs.unsqueeze(1)
        inputs = einops.repeat(inputs, "b l c h w -> b (repeat l) c h w", repeat=20)

        outputs = self.decoder(inputs, hidden_states)

        return outputs


def test():
    input_tensor = torch.randn((1, 20, 1, 480, 576))

    model = Model(
        in_chan=1, encoder_chan_rise=32, time_steps=20, generate_channels=[8, 16, 32, 128, 512],
        num_l_blocks=4, input_shape=(8, 15, 18), h_state_dims=[256, 128, 64, 32], x_dims=[512, 256, 128, 64]
    )
    output = model(input_tensor)
    print(output.shape)
    return output


if __name__ == '__main__':
    test()







