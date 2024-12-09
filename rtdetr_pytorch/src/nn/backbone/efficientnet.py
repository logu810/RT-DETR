import torch
import torch.nn as nn 
from transformers import EfficientNetModel


from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(EfficientNet, self).__init__()  
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7", in_channels=32)
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x