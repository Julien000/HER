import torch
import torch.nn.functional as F
def mask_feature(feature, mask_ratio):
    if mask_ratio <= 0:
        return feature
    
    # 获取特征形状
    batch_size, feature_dim = feature.shape[:2] if len(feature.shape) == 2 else (feature.shape[0], -1)
    
    # 创建mask (0表示保留，1表示mask掉)
    num_features = feature.shape[-1] if len(feature.shape) > 1 else 1
    mask_num = int(num_features * mask_ratio)
    
    if len(feature.shape) == 2:  # [batch_size, feature_dim]
        mask_indices = torch.rand(feature.shape[0], feature.shape[1], device=feature.device).topk(mask_num, dim=1)[1]
        mask = torch.zeros_like(feature, dtype=torch.bool)
        for i in range(feature.shape[0]):
            mask[i, mask_indices[i]] = True
        masked_feature = feature.masked_fill(mask, 0)  # 用0填充被mask的部分
        
    elif len(feature.shape) == 3:  # [batch_size, seq_len, feature_dim] 或 [batch_size, channels, features]
        # 对于序列维度进行mask
        mask_size = feature.shape[1] * feature.shape[2] if len(feature.shape) == 3 else feature.shape[1]
        total_elements = feature.shape[1] * feature.shape[2]
        mask_num = int(total_elements * mask_ratio)
        
        # 重新整形为二维进行mask
        reshaped = feature.view(feature.shape[0], -1)
        mask_positions = torch.rand(feature.shape[0], reshaped.shape[1], device=feature.device).topk(mask_num, dim=1)[1]
        
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        for i in range(feature.shape[0]):
            mask[i, mask_positions[i]] = True
        
        masked_reshaped = reshaped.masked_fill(mask, 0)
        masked_feature = masked_reshaped.view_as(feature)
    
    else:
        # 对于其他维度的特征，直接在最后一个维度进行mask
        last_dim = feature.shape[-1]
        mask_num = int(last_dim * mask_ratio)
        mask_indices = torch.rand(feature.shape[:-1] + (last_dim,), device=feature.device).topk(mask_num, dim=-1)[1]
        mask = torch.zeros_like(feature, dtype=torch.bool)
        # 广播mask到正确的形状
        mask.scatter_(-1, mask_indices, True)
        masked_feature = feature.masked_fill(mask, 0)
    
    return masked_feature

