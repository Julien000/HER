import torch
import torch.nn as nn


class ConcatenationFusionLayer(nn.Module):
    def __init__(self, dim=1):
        """
        Flexible concatenation module that can concatenate any number of tensors.
        
        Args:
            dim (int): The dimension along which tensors will be concatenated. Default: -1
        """
        super(ConcatenationFusionLayer, self).__init__()
        self.dim = dim
        
    def forward(self, image_emb, text_emb, event_i_emb, event_t_emb, flag=0):
        """
        Concatenate multiple input tensors along the specified dimension.
        
        Args:
            *inputs: Variable number of tensors to concatenate
            
        Returns:
            torch.Tensor: Concatenated tensor
        """
        return torch.cat([image_emb, text_emb, event_i_emb, event_t_emb], dim=self.dim)