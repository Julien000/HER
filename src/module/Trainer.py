
import torch.nn as nn
import torch
import time
from sklearn.metrics import  recall_score, roc_auc_score, average_precision_score
import datetime
import os

from module.comm.MASK import mask_feature
from tools.visual.TJ_TSNE import showVeracityTSNE, showVeracityTSNE_two_plots, showVeracityTSNE_wEvent, showdomainTSNE, showdomainTSNE_wEvent
from tools import metrics
from tools.logger import SimpleLogger


class BaseTrainer(nn.Module):
    def __init__(self, config, train_loader):
        super().__init__()
        
        self.model = None

        self.optimizer_1 = None
        self.optimizer_2 = None
        self.scheduler_1 = None
        self.scheduler_2 = None
        
        self.criterion = None
        self.config = config
        self.device = config.device
    
        # Storage for epoch metrics
        self.train_metrics = {
            'loss': [],
            'accuracy': [],
            'recall': [],
            'auc': [],
            'ap': []
        }
        self.val_metrics = {
            'accuracy': [],
            'recall': [],
            'auc': [],
            'ap': []
        }
        self.best_results=None
        self.tsne_feats=self.get_feats_recoder()

    def get_recoder(self):
        return {
            "logits": [],
            'probs':[],
            "labels":[]
        }

    def get_feats_recoder(self):
        return {
            "raw":[],
            "img_feats":[],
            "txt_feats":[],
            "et":[],
            "vt":[],
            "feats":[],
            "preds":[],
            "mlp": [],
            "labels":[],
            "domain_labels": [],
            "event_labels": [],
        }
    
    def save_epoch_feats(self, feats, save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        save_feats = {
            "raw": torch.stack(self.tsne_feats['raw']).float().detach().cpu(),
            "img_feats": torch.stack(self.tsne_feats['img_feats']).float().detach().cpu(),
            "txt_feats": torch.stack(self.tsne_feats['txt_feats']).float().detach().cpu(),
            "et": torch.stack(self.tsne_feats['et']).float().detach().cpu() if self.tsne_feats['et']!=[] else [],
            "vt": torch.stack(self.tsne_feats['vt']).float().detach().cpu() if self.tsne_feats['vt']!=[] else [],
            "feats": torch.stack(self.tsne_feats['feats']).float().detach().cpu(),
            "preds": torch.stack(self.tsne_feats['preds']).detach().cpu(),
            "auth_labels": torch.stack(self.tsne_feats['labels']).detach().cpu(),
            "domain_labels": torch.stack(self.tsne_feats['domain_labels']).detach().cpu(),
            "event_labels": torch.stack(self.tsne_feats['event_labels']).detach().cpu(),
        }
        torch.save(save_feats, save_dir)
        print(f"feats saving to {save_dir}")

    def share_eval(self, batch, Training=True):
        pass

    def compute_loss_optimizer(self, res):
        pass

    def train_step(self, batch, batch_idx):
        """
        Perform a single training step
        """

      
        response = self.share_eval(batch, Training=True)
        loss_res = self.compute_loss_optimizer(response )

        # Calculate metrics
        batch_label=response['targets']
        logits =  response['logits'] 
        correct = (logits == batch_label.squeeze()).sum().item()
        accuracy = correct / len(batch_label)

        loss_res['train_ACC']=accuracy
        return {**response, **loss_res}
    
    def train_epoch_end(self, epoch_metrics, epoch):
        """
        Called at the end of a training epoch
        """
        # Aggregate metrics across batches
        avg_loss = sum([m['loss'] * m['batch_size'] for m in epoch_metrics]) / \
                   sum([m['batch_size'] for m in epoch_metrics])
        
        avg_accuracy = sum([m['train_ACC'] * m['batch_size'] for m in epoch_metrics]) / \
                       sum([m['batch_size'] for m in epoch_metrics])
        
        # Collect all predictions and targets for detailed metrics
        all_preds = torch.cat([m['logits'] for m in epoch_metrics])
        all_probs = torch.cat([m['probs'] for m in epoch_metrics])
        all_targets = torch.cat([m['targets'] for m in epoch_metrics])
        
        # Calculate additional metrics
        recall = recall_score(all_targets.numpy(), all_preds.numpy())
        # Calculate AUC and AP if possible (need both classes present)
        try:
            auc = roc_auc_score(all_targets.numpy(), all_probs.numpy())
        except ValueError:
            auc = 0.0

        try:
            ap = average_precision_score(all_targets.numpy(), all_probs.numpy())
        except ValueError:
            ap = 0.0
        
        # Store epoch metrics
        self.train_metrics['loss'].append(avg_loss)
        self.train_metrics['accuracy'].append(avg_accuracy)
        self.train_metrics['recall'].append(recall)
        self.train_metrics['auc'].append(auc)
        self.train_metrics['ap'].append(ap)
        
        # Log epoch summary
        self.logger.log(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, "
                        f"Recall: {recall:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'recall': recall,
            'auc': auc,
            'ap': ap
        }
    
    def train(self, train_loader, epoch=-1):
        # Training phase
        train_step_outputs = []
        self.model.train()
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            step_metrics = self.train_step(batch, batch_idx)
            train_step_outputs.append(step_metrics)
            
            # Log progress periodically
            if (batch_idx + 1) % (len(train_loader) // 5) == 0:
                avg_loss = sum([m['loss'] * m['batch_size'] for m in train_step_outputs]) / \
                           sum([m['batch_size'] for m in train_step_outputs])
                
                avg_event_loss = sum([m['loss_event'] * m['batch_size'] for m in train_step_outputs]) / \
                                 sum([m['batch_size'] for m in train_step_outputs])
                
                avg_class_loss = sum([m['loss_classifier'] * m['batch_size'] for m in train_step_outputs]) /\
                                 sum([m['batch_size'] for m in train_step_outputs])
                
                avg_contra_loss = sum([m['loss_contra'] * m['batch_size'] for m in train_step_outputs]) / \
                                  sum([m['batch_size'] for m in train_step_outputs])
                
                self.logger.log(f"Epoch {epoch+1:04d} | Step {batch_idx+1:04d}/{len(train_loader):04d} | "
                                f"Loss_event {avg_event_loss:.4f} | Loss_class {avg_class_loss:.4f} | "
                                f"Loss_contra {avg_contra_loss:.4f} | Loss {avg_loss:.4f} | "
                                f"Time {time.time() - start_time:.4f}")
        
        epoch_time = time.time() - start_time
        
        # End of training epoch
        train_epoch_metrics = self.train_epoch_end(train_step_outputs, epoch)
        self.logger.log(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    def validation_step(self, batch, batch_idx):
        """
        Perform a single test/validation step
        """
        if self.config.args.maskrate>=0:
            if self.config.args.maskbranch=="text":
                batch['txt_feat'] = mask_feature(batch['txt_feat'], self.config.args.maskrate) 
            else:
                batch['img_feat'] = mask_feature(batch['img_feat'], self.config.args.maskrate)

        response = self.share_eval(batch, Training=False) #eval() 
        # Return step metrics
        res = {
            'batch_size': len(response['batch_label']),
            'logits': response['logits'],
            'probs': response['probs'],
            'labels': response['targets']
        }
        return res
    
    def validation_end(self, test_output):
        """
        Called at the end of testing/validation
        """
        # Aggregate metrics across batches
        preds, probs, labels = test_output['logits'], test_output['probs'], test_output['labels']
        
        res = metrics.compute_extended_metrics(preds, probs, labels)
        metrics.print_metrics_report(res, self.logger)

        return {
            'accuracy': res['ACC'],
            "res": res,
        }

    def validation(self, val_loader):
        """
        Run validation on the given data loader
        """
        test_step_outputs=self.get_recoder()
        
        for batch_idx, batch in enumerate(val_loader):
            res = self.validation_step(batch, batch_idx)
            preds, probs, labels=res['logits'], res['probs'], res['labels']
            test_step_outputs['logits'].extend(preds)
            test_step_outputs['probs'].extend(probs)
            test_step_outputs['labels'].extend(labels)
        
        # End of validation
        val_metrics = self.validation_end(test_step_outputs)
        return val_metrics
    
    def fit(self, train_loader, val_loader, epochs=100):
        """
        Main training loop
        """
        # Setup logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = SimpleLogger(f"{self.config.log_dir}/{self.config.dataset}_training_log_{timestamp}.txt") 
        self.logger.log(self.get_parameter_number(self.model))
        self.logger.log_config(self.config)

        self.logger.log(f"Starting training for {epochs} epochs")
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            self.train(train_loader, epoch)
            
            # Validation phase if provided
            if val_loader:
                self.model.eval()
                val_metrics = self.validation(val_loader)
                # Save best model
                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy'] 
                    torch.save(self.model.state_dict(), self.config.out_dir+"best_acc_ckpt.pth")
                    self.logger.log(f"New best model saved with accuracy: {best_acc:.4f}")
                    self.best_results=val_metrics['res']


                save_feats_dir=self.config.out_dir + f"save_feat/epoch_{epoch}.pt"
                self.save_epoch_feats(self.tsne_feats, save_feats_dir)
            
            self.tsne_feats=self.get_feats_recoder()

        self.logger.log("-" * 30)
        self.logger.log(self.best_results)
        self.logger.log("-" * 30)

    def load_model_weight(self, ckpt):
        state_dict=torch.load(ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        print(f"Loading model from {ckpt} succcess!!!!")

    def Testing(self, val_loader,  epochs=-1):
        """
        Main training loop
        """

        # Setup logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_info = f"MASK_{self.config.args.maskbranch}{self.config.args.maskrate}"
        self.logger = SimpleLogger(f"{self.config.log_dir}/{self.config.dataset}_{mask_info}_{timestamp}.txt") 
        self.logger.log(self.get_parameter_number(self.model))
        self.logger.log_config(self.config)

        self.logger.log(f"Starting training for {epochs} epochs")
        best_acc = 0.0

        # Validation phase if provided
        if val_loader:
            self.model.eval()
            val_metrics = self.validation(val_loader)
            
            # Save best model
            if val_metrics['accuracy'] > best_acc:
                self.best_results=val_metrics['res']

        if self.config.show_TSNE:
            #Saving faetures 
            save_feats_dir=self.config.out_dir + f"save_feat/epoch_{epochs}.pt"
            self.save_epoch_feats(self.tsne_feats, save_feats_dir)

        self.logger.log("-" * 30)
        self.logger.log(self.best_results)
        self.logger.log("-" * 30)

    def get_parameter_number(self, model):
        # Print model parameter count
        total_num = sum(p.numel() for p in model.parameters()) /1000/1000
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000/1000
        return 'Total parameters: {} M, Trainable parameters: {} M'.format(total_num, trainable_num)
