import torch
import torch.nn as nn


class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=8 * 8 * 128, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(in_features=256, out_features=10, bias=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_layers(batch))
