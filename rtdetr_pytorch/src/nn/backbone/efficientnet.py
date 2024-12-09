import torch
import torch.nn as nn 
from transformers import EfficientNetModel


from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, configuration, return_idx=[1, 2, 3]):
        super(EfficientNet, self).__init__()
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.return_idx = return_idx

        # #Map EfficientNet output channels to HybridEncoder input channels
        self.channel_mapper = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels, out_channels in zip([32, 48, 136], [192, 512, 1088])
        ])


    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        features = [outputs.hidden_states[i] for i in self.return_idx]

        # Debug: Print shapes of features before mapping
        print("EfficientNet feature shapes before mapping:")
        for i, feature in enumerate(features):
            print(f"Feature {i}: {feature.shape}")
        
        features = [self.channel_mapper[i](feature) for i, feature in enumerate(features)]
        return features

