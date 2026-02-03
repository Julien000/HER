
import torch.nn as nn
from torch.autograd import Variable, Function
import torch
import time

from transformers import AdamW, get_cosine_schedule_with_warmup

from module.loss.Triplet_new import triplet_loss_content,triplet_loss_event
from module.model.Ours import Our_Model, Our_Model_text
from tools.utils import save_images
from .Trainer import BaseTrainer


class BinaryFocalLoss(nn.Module):
    """
    Alternative implementation specifically for binary classification
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weight for positive class (0.25 is common)
            gamma (float): Focusing parameter (2.0 is common)
            reduction (str): How to reduce the loss
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions, shape (N,) or (N, 1) - logits or probabilities
            targets: Ground truth, shape (N,) - binary labels (0 or 1)
        """
        targets = targets.float()
        
        # Apply sigmoid to convert logits to probabilities
        p = torch.sigmoid(inputs)
        
        # Clamp to prevent NaN's and Inf's
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        pos_loss = -self.alpha * (1 - p) ** self.gamma * targets * torch.log(p)
        neg_loss = -(1 - self.alpha) * p ** self.gamma * (1 - targets) * torch.log(1 - p)
        
        focal_loss = pos_loss + neg_loss
        
        return focal_loss.mean()


class Our_ModelTrainer(BaseTrainer):

    def __init__(self, config, train_loader ):
        super().__init__(config, train_loader )
        
        self.model = Our_Model(config)
        self.model.to(config.device)
        params_1 = self.model.parameters()
        self.optimizer_1 = AdamW(params_1, lr=2e-5, weight_decay=1e-4)
        self.scheduler_1 = get_cosine_schedule_with_warmup(self.optimizer_1, num_warmup_steps=len(train_loader), num_training_steps=config.epochs * len(train_loader))


        params_2 =  self.model.adv_params_2()
        self.optimizer_2 = AdamW(params_2, lr=2e-5, weight_decay=1e-4)
        # self.optimizer_2 = AdamW(params_2, lr=1e-6, weight_decay=1e-4)
        self.scheduler_2 = get_cosine_schedule_with_warmup(self.optimizer_2, num_warmup_steps=len(train_loader), num_training_steps=config.epochs * len(train_loader))

        # self.criterion =  BinaryFocalLoss()
        self.criterion =  nn.CrossEntropyLoss()
        self.device = config.device
        self.config = config
        self.model.to(self.device)
        
    def share_eval(self, batch, Training=True):
        batch_text, batch_image, batch_event, batch_label, batch_domain = \
        batch['txt_feat'], batch['img_feat'], batch['event'],  batch['label'], batch['domain']
        batch_idx= batch['idx']
        
    
        
        batch_text   = batch_text.to(self.device)
        batch_image  = batch_image.to(self.device)
        batch_label  = Variable(batch_label.to(self.device))
        batch_event  = Variable(batch_event.to(self.device))
        batch_domain = Variable(batch_domain.to(self.device))

        if self.config.args.maskrate != 0.0:
            batch_text   = batch_text.to(self.device)
            batch_image  = batch_image.to(self.device)

        shuffle_num, eventOut_t, contentOut_t, eventOut_i, contentOut_i=None,None,None,None,None
        event_t, content_t, content_i, event_i= None, None, None, None

        if Training:
            # shuffle_num, content_pred, content, eventOut_t, contentOut_t, eventOut_i, contentOut_i, content_t, event_t, content_i, event_i = self.model(
            #     batch_text, batch_image)
            shuffle_num, content_pred, content, eventOut_t, contentOut_t, eventOut_i, contentOut_i, content_t, event_t, content_i, event_i = self.model(
                batch_text, batch_image)
        else:
            # content_pred, content, content_t, event_t, content_i, event_i = self.model.test_struct(batch_text, batch_image)
            content_pred, content, content_t, event_t, content_i, event_i = self.model.test_struct(batch_text, batch_image, batch_idx)
        
        bs, *_=batch_image.shape
        probs = torch.softmax(content_pred, dim=1)[:, 1]  # Probability of positive class
        logits = torch.argmax(content_pred, dim=1)
        # record information
        self.tsne_feats['raw'].extend(torch.cat([batch_image.view(bs,-1), batch_text.view(bs,-1)], dim=1))
        self.tsne_feats['img_feats'].extend(batch_image.view(bs,-1))
        self.tsne_feats['txt_feats'].extend(batch_text.view(bs,-1))
        self.tsne_feats['feats'].extend(content)
        self.tsne_feats['preds'].extend(content_pred)
        self.tsne_feats['labels'].extend(batch_label)
        self.tsne_feats['et'].extend(event_t)
        self.tsne_feats['vt'].extend(content_t)
        self.tsne_feats['domain_labels'].extend(batch_domain)
        self.tsne_feats['event_labels'].extend(batch_event)

        return {
            "content_pred": content_pred,
            "shuffle_num": shuffle_num, 
            "content": content, 
            "eventOut_t": eventOut_t , 
            "contentOut_t": contentOut_t , 
            "eventOut_i": eventOut_i ,
            "contentOut_i": contentOut_i , 
            "content_t": content_t, 
            "event_t": event_t, 
            "content_i": content_i, 
            "event_i": event_i,
            "batch_label": batch_label,
            "batch_event": batch_event,
            "batch_domain": batch_domain,
            "logits": logits.detach().cpu(),
            "probs": probs.detach().cpu(),
            "targets": batch_label.cpu(),
        }

    def compute_loss_optimizer(self, res):
        shuffle_num = res['shuffle_num']
        content_pred, batch_label= res['content_pred'],res['batch_label']
        contentOut_t, contentOut_i= res['contentOut_t'],res['contentOut_i']
        content_t, content_i= res['content_t'],res['content_i']
        event_t, event_i, batch_event= res['event_t'],res['event_i'], res['batch_event']
        eventOut_t, eventOut_i, batch_domain = res['eventOut_t'],res['eventOut_i'], res['batch_domain']

        # Update schedulers
        if self.scheduler_1:
            self.scheduler_1.step()
        if self.scheduler_2:
            self.scheduler_2.step()
        
        # Calculate losses
        loss_event = (self.criterion(eventOut_t, batch_event.repeat_interleave(shuffle_num))) + \
                     (self.criterion(eventOut_i, batch_event.repeat_interleave(shuffle_num)))
        
        loss_classfier = self.criterion(content_pred, batch_label)
        
        loss_authenticity = (self.criterion(contentOut_t, batch_label.repeat_interleave(shuffle_num))) + \
                            (self.criterion(contentOut_i, batch_label.repeat_interleave(shuffle_num)))
        
        loss_contra = triplet_loss_content(content_t, batch_label, batch_domain, batch_event) + \
                      triplet_loss_content(content_i, batch_label, batch_domain, batch_event) + \
                      triplet_loss_event(event_i, batch_domain, batch_event) + \
                      triplet_loss_event(event_t, batch_domain, batch_event)
        
        loss_authenticity_pure = self.criterion(content_t, batch_label) + self.criterion(content_i, batch_label)
        loss_event_pure = self.criterion(event_t, batch_event) + self.criterion(event_i, batch_event)
        
        # First optimization step
        if self.optimizer_1:
            self.optimizer_1.zero_grad()
            loss_1 = 0.6 * (loss_authenticity + loss_event) + loss_contra + loss_classfier
            loss_1.backward(retain_graph=True)
            self.optimizer_1.step()
        
        # Second optimization step
        if self.optimizer_2:
            self.optimizer_2.zero_grad()
            loss = loss_authenticity_pure + loss_event_pure
            loss.backward()
            self.optimizer_2.step()
        
        step_metrics = {
            'loss': loss.item(),
            'loss_event': loss_event.item(),
            'loss_classifier': loss_classfier.item(),
            'loss_contra': loss_contra.item(),
            'batch_size': batch_label.shape[0],
        }

        return step_metrics


class TextModelrainer(Our_ModelTrainer):
    """
        A simple PyTorch model wrapper that provides train_step, test_step,
        train_epoch_end, and test_end methods without using third-party libraries.
    """
    def __init__(self, config, train_loader ):
        super().__init__(config, train_loader )
    
        self.model = Our_Model_text(config)
        self.model.to(config.device) 
        params_1 = self.model.parameters()
        self.optimizer_1 = AdamW(params_1, lr=2e-5, weight_decay=1e-4)
        self.scheduler_1 = get_cosine_schedule_with_warmup(self.optimizer_1, num_warmup_steps=len(train_loader), num_training_steps=config.epochs * len(train_loader))


        params_2 =  self.model.adv_params_2()
        self.optimizer_2 = AdamW(params_2, lr=2e-5, weight_decay=1e-4)
        # self.optimizer_2 = AdamW(params_2, lr=1e-6, weight_decay=1e-4)
        self.scheduler_2 = get_cosine_schedule_with_warmup(self.optimizer_2, num_warmup_steps=len(train_loader), num_training_steps=config.epochs * len(train_loader))

        # self.criterion =  BinaryFocalLoss()
        self.criterion =  nn.CrossEntropyLoss()
        self.device = config.device
        self.config = config
        self.model.to(self.device)

    def compute_loss_optimizer(self, res):
        shuffle_num = res['shuffle_num']
        content_pred, batch_label= res['content_pred'],res['batch_label']
        contentOut_t, contentOut_i= res['contentOut_t'],res['contentOut_i']
        content_t, content_i= res['content_t'],res['content_i']
        event_t, event_i, batch_event= res['event_t'],res['event_i'], res['batch_event']
        eventOut_t, eventOut_i, batch_domain = res['eventOut_t'],res['eventOut_i'], res['batch_domain']

        # Update schedulers
        if self.scheduler_1:
            self.scheduler_1.step()
        if self.scheduler_2:
            self.scheduler_2.step()
        
        # Calculate losses
        loss_event = (self.criterion(eventOut_t, batch_event.repeat_interleave(shuffle_num))) 
        loss_classfier = self.criterion(content_pred, batch_label)
        
        loss_authenticity = (self.criterion(contentOut_t, batch_label.repeat_interleave(shuffle_num))) 
        
        loss_contra = triplet_loss_content(content_t, batch_label, batch_domain, batch_event) + \
                      triplet_loss_event(event_t, batch_domain, batch_event)
        
        loss_authenticity_pure = self.criterion(content_t, batch_label)  
        loss_event_pure = self.criterion(event_t, batch_event)  
        
        # First optimization step
        if self.optimizer_1:
            self.optimizer_1.zero_grad()
            loss_1 = 0.6 * (loss_authenticity + loss_event) + loss_contra + loss_classfier
            loss_1.backward(retain_graph=True)
            self.optimizer_1.step()
        
        # Second optimization step
        if self.optimizer_2:
            self.optimizer_2.zero_grad()
            loss = loss_authenticity_pure + loss_event_pure
            loss.backward()
            self.optimizer_2.step()
        
        step_metrics = {
            'loss': loss.item(),
            'loss_event': loss_event.item(),
            'loss_classifier': loss_classfier.item(),
            'loss_contra': loss_contra.item(),
            'batch_size': batch_label.shape[0],
        }

        return step_metrics
    
    def share_eval(self, batch, Training=True):
        batch_text, batch_image, batch_event, batch_label, batch_domain = \
        batch['txt_feat'], batch['img_feat'], batch['event'],  batch['label'], batch['domain']
        
        batch_text = batch_text.to(self.device)
        batch_image = batch_image.to(self.device)
        batch_label = Variable(batch_label.to(self.device))
        batch_event = Variable(batch_event.to(self.device))
        batch_domain = Variable(batch_domain.to(self.device))

        shuffle_num, eventOut_t, contentOut_t, eventOut_i, contentOut_i=None,None,None,None,None
        event_t, content_t, content_i, event_i= None, None, None, None

        if Training:
            shuffle_num, content_pred, content, eventOut_t, contentOut_t, eventOut_i, contentOut_i, content_t, event_t, content_i, event_i = self.model(
                batch_text, batch_image)
        else:
            content_pred, content, content_t, event_t, content_i, event_i = self.model.test_struct(batch_text, batch_image)
        
        bs, *_=batch_image.shape
        probs = torch.softmax(content_pred, dim=1)[:, 1]  # Probability of positive class
        logits = torch.argmax(content_pred, dim=1)
        # record information
        self.tsne_feats['raw'].extend(torch.cat([batch_image.view(bs,-1), batch_text.view(bs,-1)], dim=1))
        self.tsne_feats['img_feats'].extend(batch_image.view(bs,-1))
        self.tsne_feats['txt_feats'].extend(batch_text.view(bs,-1))
        self.tsne_feats['feats'].extend(content)
        self.tsne_feats['preds'].extend(content_pred)
        self.tsne_feats['labels'].extend(batch_label)
        self.tsne_feats['et'].extend(event_t)
        self.tsne_feats['vt'].extend(content_t)
        self.tsne_feats['domain_labels'].extend(batch_domain)
        self.tsne_feats['event_labels'].extend(batch_event)

        return {
            "content_pred": content_pred,
            "shuffle_num": shuffle_num, 
            "content": content, 
            "eventOut_t": eventOut_t , 
            "contentOut_t": contentOut_t , 
            "eventOut_i": eventOut_i ,
            "contentOut_i": contentOut_i , 
            "content_t": content_t, 
            "event_t": event_t, 
            "content_i": content_i, 
            "event_i": event_i,
            "batch_label": batch_label,
            "batch_event": batch_event,
            "batch_domain": batch_domain,
            "logits": logits.detach().cpu(),
            "probs": probs.detach().cpu(),
            "targets": batch_label.cpu(),
        }   

class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


class ReverseLayerF(Function):

    def forward(self, x, args):
        self.lambd = args.lambd
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return Exp.apply(x)

