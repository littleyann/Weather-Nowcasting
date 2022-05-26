import einops
import torch
from torch import nn
import torch.nn.functional as F


class GBlock(nn.Module):
    def __init__(self, in_chan, out_chan, up):
        super(GBlock, self).__init__()
        self.up = up
        self.in_chan = in_chan
        self.out_chan = out_chan

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_chan, in_chan, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(1, 1))

        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_chan)
        self.bn2 = nn.BatchNorm2d(in_chan)
        if self.up is True:
            self.up_sample = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.up is True:
            out1 = self.up_sample(x)
            out1 = self.conv1(out1)
        else:
            out1 = self.conv1(x)

        out2 = self.bn1(x)
        out2 = self.act(out2)
        if self.up is True:
            out2 = self.up_sample(out2)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.act(out2)
        out2 = self.conv3(out2)

        return out1 + out2


class DBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=(1, 1))

        self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.down_sample(out1)

        out2 = self.act(x)
        out2 = self.conv2(out2)
        out2 = self.act(out2)
        out2 = self.conv3(out2)
        out2 = self.down_sample(out2)

        return out1 + out2
    
    
class LBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(LBlock, self).__init__()

        self.path1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_chan, in_chan, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.path2 = nn.Conv2d(in_chan, out_chan-in_chan, kernel_size=(1, 1))

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out2 = torch.cat([x, out2], dim=1)
        return out1 + out2







