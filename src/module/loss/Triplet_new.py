import torch
import torch.nn.functional as F


def triplet_loss_content(r, Batch_label, Batch_domain, Batch_event, margin=0.5):
    # 计算标签相等和域不相等的掩码
    labels_eq = Batch_label.unsqueeze(1) == Batch_label.unsqueeze(0)
    domains_neq = Batch_domain.unsqueeze(1) != Batch_domain.unsqueeze(0)

    # 正样本：batch_label相同，batch_domain不同，batch_event不同
    positive_mask = labels_eq & domains_neq & (Batch_event.unsqueeze(1) != Batch_event.unsqueeze(0))

    # 负样本：batch_label不同，Batch_event相同
    # negative_mask = ~labels_eq & (Batch_event.unsqueeze(1) == Batch_event.unsqueeze(0))
    negative_mask = ~labels_eq


    # 找到三元组掩码
    triplets_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)

    # 找到所有有效三元组的索引
    triplets_indices = triplets_mask.nonzero(as_tuple=True)

    # 如果没有找到有效的三元组，返回零损失
    if triplets_indices[0].numel() == 0:
        return torch.tensor(0.0, device=r.device)

    # 根据索引提取锚点、正样本和负样本
    anchors = r[triplets_indices[0]]
    positives = r[triplets_indices[1]]
    negatives = r[triplets_indices[2]]

    # 计算三元组损失
    loss = F.triplet_margin_loss(anchors, positives, negatives, margin=0.5)
    return loss


def triplet_loss_event(r, Batch_domain, Batch_event, margin=0.5):

    # 计算domain相同，事件不同
    domain_eq = Batch_domain.unsqueeze(1) == Batch_domain.unsqueeze(0)
    event_neq = Batch_event.unsqueeze(1) != Batch_event.unsqueeze(0)

    # 正样本：batch_domain相同，batch_event不同
    positive_mask = domain_eq & event_neq

    # 负样本：domain不同，event不同
    negative_mask = ~domain_eq & event_neq

    # 找到三元组掩码
    triplets_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)

    # 找到所有有效三元组的索引
    triplets_indices = triplets_mask.nonzero(as_tuple=True)

    # 如果没有找到有效的三元组，返回零损失
    if triplets_indices[0].numel() == 0:
        return torch.tensor(0.0, device=r.device)

    # 根据索引提取锚点、正样本和负样本
    anchors = r[triplets_indices[0]]
    positives = r[triplets_indices[1]]
    negatives = r[triplets_indices[2]]

    # 计算三元组损失
    loss = F.triplet_margin_loss(anchors, positives, negatives, margin=0.5)
    return loss