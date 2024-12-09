import torch
import torch.nn as nn 
from transformers import MobileNetV2Model


from src.core import register

__all__ = ['MobileNet']

@register
class MobileNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(MobileNet, self).__init__()  
        self.model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x