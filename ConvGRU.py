import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
        :param input_dim: input information channel
        :param hidden_dim: hidden state information channel
        :param kernel_size: kernel size
        """
        super(ConvGRUCell, self).__init__()
        self.sigmoid = nn.Sigmoid()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, stride=(1, 1))
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, stride=(1, 1))
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, stride=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, x, previous_state):
        z = torch.cat([x, previous_state], dim=1)
        reset = self.sigmoid(self.reset_gate(z))
        update = self.sigmoid(self.update_gate(z))

        new_hidden_state_tilda = self.tanh(self.out_gate(torch.cat([x, previous_state * reset], dim=1)))
        new_hidden_state = (1 - update) * previous_state + update * new_hidden_state_tilda

        return new_hidden_state, new_hidden_state


class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.gru_cell = ConvGRUCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x, hidden_state):
        predict_states = []
        states_len = x.size(1)
        for step in range(states_len):
            state_input = x[:, step, :, :, :]
            state_output, hidden_state = self.gru_cell(state_input, hidden_state)
            predict_states.append(state_output)

        # predict_states = torch.stack(predict_states, dim=1)
        return predict_states


def test():
    hidden = torch.rand([2, 256, 8, 8], dtype=torch.float32)

    x = torch.rand([2, 20, 512, 8, 8], dtype=torch.float32)

    model = ConvGRU(input_dim=512, hidden_dim=256, kernel_size=(3, 3))

    predict_states = model(x, hidden)
    print('predict_states shape: {}'.format(predict_states[0].shape))
    return predict_states


if __name__ == '__main__':
    states = test()
