import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, h_state_dim, x_dim, kernel_size):
        """
        :param h_state_dim: input information channel
        :param x_dim: hidden state information channel
        :param kernel_size: kernel size
        """
        super(ConvGRUCell, self).__init__()
        self.sigmoid = nn.Sigmoid()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.reset_gate = nn.Conv2d(h_state_dim + x_dim, h_state_dim, kernel_size, padding=padding, stride=(1, 1))
        self.update_gate = nn.Conv2d(h_state_dim + x_dim, h_state_dim, kernel_size, padding=padding, stride=(1, 1))
        self.out_gate = nn.Conv2d(h_state_dim + x_dim, h_state_dim, kernel_size, padding=padding, stride=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state):
        z = torch.cat([x, hidden_state], dim=1)
        reset = self.sigmoid(self.reset_gate(z))
        update = self.sigmoid(self.update_gate(z))

        new_hidden_state_tilda = self.tanh(self.out_gate(torch.cat([x, hidden_state * reset], dim=1)))
        new_hidden_state = (1 - update) * hidden_state + update * new_hidden_state_tilda

        return new_hidden_state


class ConvGRU(nn.Module):
    def __init__(self, h_state_dim, x_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.gru_cell = ConvGRUCell(h_state_dim, x_dim, kernel_size)

    def forward(self, x, hidden_state):
        predicts = []
        time_steps = x.size(1)
        for step in range(time_steps):
            state_input = x[:, step, :, :, :]
            hidden_state = self.gru_cell(state_input, hidden_state)
            predicts.append(hidden_state)

        return predicts


def test():
    hidden_state = torch.rand([2, 256, 8, 8], dtype=torch.float32)
    x = torch.rand([2, 20, 512, 8, 8], dtype=torch.float32)
    model = ConvGRU(h_state_dim=256, x_dim=512, kernel_size=(3, 3))

    predict_states = model(x, hidden_state)
    return predict_states


if __name__ == '__main__':
    states = test()
