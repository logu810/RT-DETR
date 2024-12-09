import torch
import torch.nn as nn 
from transformers import EfficientNetModel


from src.core import register

__all__ = ['EfficientNet']

@register
class EfficientNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(EfficientNet, self).__init__()  
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.return_idx = return_idx

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        x = [outputs.hidden_states[i] for i in self.return_idx]  # Return layers at return_idx
        # Add debugging print
        print([x.shape for x in [outputs.hidden_states[i] for i in self.return_idx]])

        return x