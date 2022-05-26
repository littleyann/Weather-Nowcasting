import torch
from torch import nn
from Blocks import LBlock


class Input(nn.Module):
    def __init__(self, generate_channels, num, shape):
        super(Input, self).__init__()
        self.shape = shape
        self.norm = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.conv = nn.Conv2d(generate_channels[0], generate_channels[0], kernel_size=(3, 3), padding=(1, 1))
        self.blocks = nn.Sequential(*[LBlock(generate_channels[i], generate_channels[i+1]) for i in range(num)])

    def forward(self, x):
        z = self.norm.sample(self.shape)
        z = z.permute(3, 0, 1, 2).type_as(x)
        out = self.conv(z)
        out = self.blocks(out)
        return out


if __name__ == '__main__':
    model = Input([8, 16, 32, 128, 512], 4, (8, 8, 8)).cuda()
    output_tensor = model()
    print(output_tensor.shape)
