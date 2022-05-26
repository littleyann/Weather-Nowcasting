import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class SDRLoss(nn.Module):
    def __init__(self):
        super(SDRLoss, self).__init__()

    def loss(self, x, y, x_norm, y_norm):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        output = torch.bmm(y.view(y.shape[0], 1, y.shape[-1]), x.view(x.shape[0], x.shape[-1], 1)).reshape(-1) / \
                 (x_norm * y_norm + 1e-8)
        return output

    def forward(self, noisy, clean, enhance):
        z = noisy - clean
        z_hat = noisy - enhance

        clean_norm = torch.norm(clean, p=2, dim=-1, keepdim=True).squeeze(1)
        enhance_norm = torch.norm(enhance, p=2, dim=-1, keepdim=True).squeeze(1)

        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True).squeeze(1)
        z_hat_norm = torch.norm(z_hat, p=2, dim=-1, keepdim=True).squeeze(1)

        alpha = (clean_norm ** 2) / (clean_norm ** 2 + z_norm ** 2 + 1e-8)
        sdr_loss = -alpha * self.loss(clean, enhance, clean_norm, enhance_norm) - \
                   (1 - alpha) * self.loss(z, z_hat, z_norm, z_hat_norm)
        return sdr_loss.mean()


class ClipSDRLoss(nn.Module):
    def __init__(self):
        super(ClipSDRLoss, self).__init__()

    def sdr(self, x, y):
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).squeeze(1)
        x_y_norm = torch.norm(x - y, p=2, dim=-1, keepdim=True).squeeze(1)
        output = 10 * torch.log10(x_norm ** 2 / (x_y_norm ** 2 + 1e-6))
        return output

    def clip(self, x):
        beta = 20
        output = beta * torch.tanh(x / beta)
        return output

    def forward(self, noisy, clean, enhance):
        z = noisy - clean
        z_hat = noisy - enhance

        loss = -0.5 * (self.clip(self.sdr(clean, enhance)) + self.clip(self.sdr(z, z_hat)))

        return loss.mean()