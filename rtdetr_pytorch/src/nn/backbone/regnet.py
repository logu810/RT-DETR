import torch
import torch.nn as nn 
from transformers import RegNetModel
from src.core import register

__all__ = ['RegNet']

@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()  
        self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
        self.return_idx = return_idx

    def forward(self, x):
        # Forward pass through the model
        outputs = self.model(x, output_hidden_states=True)
        
        # Debugging: Print the shapes of all hidden states
        print("Hidden states from the RegNet model:")
        for i, hidden_state in enumerate(outputs.hidden_states):
            print(f"Hidden state {i} shape: {hidden_state.shape}")
        
        # Select the relevant hidden states based on return_idx
        x = outputs.hidden_states[2:5]
        
        # Debugging: Print the selected feature shapes
        print("Selected hidden states (indices 2:5):")
        for i, hidden_state in enumerate(x):
            print(f"Selected Hidden state {i} shape: {hidden_state.shape}")
        
        return x
