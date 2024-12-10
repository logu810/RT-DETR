import torch
import torch.nn as nn 
from transformers import MobileNetV2Model
from src.core import register

__all__ = ['MobileNet']

@register
class MobileNet(nn.Module):
    def __init__(self, configuration, return_idx=[1, 2, 3]):
        super(MobileNet, self).__init__()  
        # Load the pretrained MobileNetV2 model
        self.model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        
        # The indices of the hidden states to be returned, as per the config
        self.return_idx = return_idx

    def forward(self, x):
        # Forward pass through the model
        outputs = self.model(x, output_hidden_states=True)
        

        # Select the relevant hidden states based on return_idx
        x = outputs.hidden_states[2:3]  # For 224 channels
        x += outputs.hidden_states[5:6]  # For 384 channels
        x += outputs.hidden_states[12:13]  # For 640 channels

        

        return x