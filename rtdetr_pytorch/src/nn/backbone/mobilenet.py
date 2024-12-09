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
        # Forward pass through the model
        outputs = self.model(x, output_hidden_states=True)
        
        # Debugging: Print the shapes of all hidden states
        print("Hidden states from the RegNet model:")
        for i, hidden_state in enumerate(outputs.hidden_states):
            print(f"Hidden state {i} shape: {hidden_state.shape}")
        
        # Select the relevant hidden states based on return_idx
        x = outputs.hidden_states[2:5] # For 640 channels

        
        # Debugging: Print the selected feature shapes
        print("Selected hidden states (indices 2:5):")
        for i, hidden_state in enumerate(x):
            print(f"Selected Hidden state {i} shape: {hidden_state.shape}")
        
        return x