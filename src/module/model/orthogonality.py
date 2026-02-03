
import torch
import torch.nn as nn

class selfAttention(nn.Module) :
    def __init__(self):
      super(selfAttention, self).__init__()
      # self.reference = nn.Sequential(
      #   nn.Linear(128, 128),
      #   nn.ReLU(),
      #   nn.Linear(128, 128)
      # )
      self.reference = nn.Sequential()
      self.reference.add_module('txt_fc1', nn.Linear(128, 128))
      self.reference.add_module('txt_dropout', nn.ReLU())
      self.reference.add_module('txt_fc2', nn.Linear(128, 128))
      
    def forward(self, A, B):
      # A_star = self.reference(A)
      A_star = B
      # A_star = torch.matmul(affinity, A) #[B, D]
      A_pos = ((A * A_star).sum(dim=1) / (A_star * A_star).sum(dim=1)).unsqueeze(1) * A_star
      A_neg = A - A_pos
      return A_pos, A_neg




