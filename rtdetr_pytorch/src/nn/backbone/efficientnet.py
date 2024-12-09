import torch
import torch.nn as nn 
from transformers import EfficientNetConfig, EfficientNetModel


from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, configuration=None, return_idx=[0, 1, 2, 3]):
        super(EfficientNet, self).__init__()
        
        # Use provided configuration or default pretrained
        if configuration is None:
            configuration = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
        
        self.model = EfficientNetModel(configuration)
        self.return_idx = return_idx

    def forward(self, x):
        # Pass input through the EfficientNet model
        outputs = self.model(x, output_hidden_states=True)
        
        # Select hidden states based on the return_idx
        x = [outputs.hidden_states[i] for i in self.return_idx]
        return x
