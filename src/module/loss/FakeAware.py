import torch 
import torch.nn.functional as F
from torch import nn

class FakeAwareAsymmetricLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_fn=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_fn = lambda_fn
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: [B, 2]  -> [logit_true, logit_fake]
        targets: [B]    -> 0=true, 1=fake
        """
        # ===== 基础 CE =====
        ce_loss = self.ce(logits, targets)

        # ===== fake-aware margin loss =====
        z_true = logits[:, 0]
        z_fake = logits[:, 1]

        # fake mask
        fake_mask = (targets == 1)

        if fake_mask.sum() > 0:
            margin_violation = self.margin - (z_fake - z_true)
            fn_loss = F.relu(margin_violation)
            fn_loss = fn_loss[fake_mask].mean()
        else:
            fn_loss = torch.tensor(0.0, device=logits.device)

        total_loss = ce_loss + self.lambda_fn * fn_loss
        return total_loss