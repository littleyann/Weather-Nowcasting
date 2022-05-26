import torch
import numpy as np


input_tensor = torch.randn((2, 16, 480, 560))
conv_3D = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
output_tensor = conv_3D(input_tensor)
print(output_tensor.shape)
print(len(input_tensor))

x = np.linspace(1, 10)