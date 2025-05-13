import torch
import torch.nn as nn


class DeepESD(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        input_channels: int,
        output_channels: int,
    ):
        super(DeepESD, self).__init__()
        self.output_shape = output_shape  # Store as attribute
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, output_channels, kernel_size=3, padding=1)

        in_features = input_shape[0] * input_shape[1] * output_channels
        out_features = output_shape[0] * output_shape[1] * output_channels
        self.out = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x.view(
            -1, self.output_channels, self.output_shape[0], self.output_shape[1]
        )
