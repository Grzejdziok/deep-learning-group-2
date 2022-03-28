import torch
import torch.nn as nn


class Lenet300100(nn.Module):

    def __init__(self):
        super(Lenet300100, self).__init__()
        # This might be an interesting case for our blogpost, because information in the Lottery Tickets paper is not
        # fully clear. For example, it is not clear whether the authors of the Lottery Ticket paper use ReLU or Sigmoid
        # as activation function. The original paper defining this network uses Sigmoid, but it seems to be slightly off
        # of the Figure 3. plot given in the Lottery Tickets paper. My bet is they are using ReLU without stating it
        # explicitly

        self.layers = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=300, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layers(batch)
