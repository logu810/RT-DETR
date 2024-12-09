import torch
import torch.nn as nn 
from transformers import EfficientNetModel


from src.core import register

__all__ = ['EfficientNet']

class EfficientNet(nn.Module):
    def __init__(self, configuration, return_idx=[1, 2, 3]):
        super(EfficientNet, self).__init__()
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.return_idx = return_idx

        # Add 1x1 convolutions to align output channels
        self.channel_mapper = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels, out_channels in zip([32, 48, 136], [192, 512, 1088])
        ])

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        features = [outputs.hidden_states[i] for i in self.return_idx]

        # Map features to the correct channels
        features = [self.channel_mapper[i](feature) for i, feature in enumerate(features)]
        return features
