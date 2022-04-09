import torch
import torch.nn as nn


class Lenet300100(nn.Module):

    def __init__(self):
        super(Lenet300100, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=300, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=300, out_features=100, bias=True),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Linear(in_features=100, out_features=10, bias=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_layers(batch))
