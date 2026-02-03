import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, precision_recall_curve
import math


def compute_extended_metrics(preds, probs, labels):
    """
    Compute extended evaluation metrics for binary fake news detection
    
    Args:
        preds: predicted class labels (0 or 1)
        probs: predicted probabilities for positive class
        labels: true labels (0 or 1)
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays if they're tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Basic metrics you already have
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Per-class F1 scores (assuming 0=real, 1=fake)
    f1_per_class = f1_score(labels, preds, labels=[0, 1], average=None, zero_division=0)
    real_f1 = f1_per_class[0]  # F1 for real news (class 0)
    fake_f1 = f1_per_class[1]  # F1 for fake news (class 1)
    
    # Macro F1 (unweighted mean of per-class F1 scores)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Handle case where we might not have both classes in predictions
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    
    try:
        ap = average_precision_score(labels, probs)
    except ValueError:
        ap = 0.0
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    mcc = matthews_corrcoef(labels, preds)  # Matthews Correlation Coefficient
    kappa = cohen_kappa_score(labels, preds)  # Cohen's Kappa
    
    # Precision-Recall curve based metrics
    try:
        precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probs)
        # Area under P-R curve
        auprc = np.trapz(precision_curve, recall_curve)
    except:
        auprc = 0.0
    
    # FPR and TPR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tpr = recall  # True Positive Rate (same as recall)
    
    # FNR and TNR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    tnr = specificity  # True Negative Rate (same as specificity)
    
    # Balanced accuracy
    balanced_acc = (tpr + tnr) / 2
    
    # Diagnostic odds ratio
    dor = (tp/fp)/(fn/tn) if fp > 0 and fn > 0 and tn > 0 else 0
    
    # Threat score / Critical success index
    ts = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
    # F2 Score (beta=2, emphasizes recall)
    f2 = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall) if (precision + recall) > 0 else 0
    
    # Jaccard index
    jaccard = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
    return {
        # Original metrics
        'ACC': accuracy,
        'AP': ap,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        
        # New metrics
        'Macro_F1': macro_f1,
        'Real_F1': real_f1,
        'Fake_F1': fake_f1,
        
        # Extended metrics
        'Specificity': specificity,
        'MCC': mcc,
        'Kappa': kappa,
        'AUPRC': auprc,
        'FPR': fpr,
        'TPR': tpr,
        'FNR': fnr,
        'TNR': tnr,
        'Balanced_ACC': balanced_acc,
        'DOR': dor,
        'Threat_Score': ts,
        'F2_Score': f2,
        'Jaccard': jaccard,
        
        # Confusion matrix components
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

def print_metrics_report(metrics_dict, logger):
    """
    Print a formatted report of the computed metrics
    
    Args:
        metrics_dict: Dictionary returned by compute_extended_metrics
    """
    logger.log("\n" + "="*60)
    logger.log("FAKE NEWS DETECTION EVALUATION REPORT")
    logger.log("="*60)
    
    # Main performance metrics
    logger.log("\nMain Performance Metrics:")
    logger.log("-"*30)
    logger.log(f"Accuracy     : {metrics_dict['ACC']:.4f}")
    logger.log(f"Precision    : {metrics_dict['Precision']:.4f}")
    logger.log(f"Recall       : {metrics_dict['Recall']:.4f}")
    logger.log(f"F1-Score     : {metrics_dict['F1']:.4f}")
    logger.log(f"Macro F1     : {metrics_dict['Macro_F1']:.4f}")
    logger.log(f"AUC          : {metrics_dict['AUC']:.4f}")
    logger.log(f"AP           : {metrics_dict['AP']:.4f}")
    logger.log(f"AUPRC        : {metrics_dict['AUPRC']:.4f}")
    
    # Per-class F1 scores
    logger.log("\nPer-Class F1 Scores:")
    logger.log("-"*30)
    logger.log(f"Real News F1 : {metrics_dict['Real_F1']:.4f}")
    logger.log(f"Fake News F1 : {metrics_dict['Fake_F1']:.4f}")
    
    # Advanced metrics
    logger.log("\nAdvanced Metrics:")
    logger.log("-"*30)
    logger.log(f"Balanced ACC : {metrics_dict['Balanced_ACC']:.4f}")
    logger.log(f"MCC          : {metrics_dict['MCC']:.4f}")
    logger.log(f"Kappa        : {metrics_dict['Kappa']:.4f}")
    logger.log(f"Specificity  : {metrics_dict['Specificity']:.4f}")
    logger.log(f"F2-Score     : {metrics_dict['F2_Score']:.4f}")
    
    # Error rates
    logger.log("\nError Rates:")
    logger.log("-"*30)
    logger.log(f"FPR (False Positive Rate) : {metrics_dict['FPR']:.4f}")
    logger.log(f"FNR (False Negative Rate) : {metrics_dict['FNR']:.4f}")
    
    # Confusion Matrix
    logger.log("\nConfusion Matrix:")
    logger.log("-"*30)
    logger.log(f"True Positives  (TP): {metrics_dict['TP']}")
    logger.log(f"True Negatives  (TN): {metrics_dict['TN']}")
    logger.log(f"False Positives (FP): {metrics_dict['FP']}")
    logger.log(f"False Negatives (FN): {metrics_dict['FN']}")
    
    logger.log("="*60)

# Extended Classification Metrics:

# Specificity (True Negative Rate)
# Matthews Correlation Coefficient (MCC)
# Cohen's Kappa
# Area Under Precision-Recall Curve (AUPRC)
# Balanced Accuracy
# Error Analysis Metrics:

# False Positive Rate (FPR)
# False Negative Rate (FNR)
# True Positive Rate (TPR)
# True Negative Rate (TNR)
# Specialized Metrics:

# Diagnostic Odds Ratio (DOR)
# Threat Score/Critical Success Index
# F2-Score (emphasizes recall)
# Jaccard Index
# Confusion Matrix Components:

# True Positives (TP)
# True Negatives (TN)
# False Positives (FP)
# False Negatives (FN)
# These additional metrics provid
# def compute_extended_metrics(preds, probs, labels):
#     """
#     Compute extended evaluation metrics for binary fake news detection
    
#     Args:
#         preds: predicted class labels (0 or 1)
#         probs: predicted probabilities for positive class
#         labels: true labels (0 or 1)
        
#     Returns:
#         dict: Dictionary containing various evaluation metrics
#     """
#     # Convert to numpy arrays if they're tensors
#     if isinstance(preds, torch.Tensor):
#         preds = preds.cpu().numpy()
#     if isinstance(probs, torch.Tensor):
#         probs = probs.cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu().numpy()
    
#     # Basic metrics you already have
#     accuracy = accuracy_score(labels, preds)
#     precision = precision_score(labels, preds, zero_division=0)
#     recall = recall_score(labels, preds, zero_division=0)
#     f1 = f1_score(labels, preds, zero_division=0)
    
#     # Handle case where we might not have both classes in predictions
#     try:
#         auc = roc_auc_score(labels, probs)
#     except ValueError:
#         auc = 0.0
    
#     try:
#         ap = average_precision_score(labels, probs)
#     except ValueError:
#         ap = 0.0
    
#     # Extended metrics
#     # Confusion matrix components
#     tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
#     # Additional metrics
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
#     mcc = matthews_corrcoef(labels, preds)  # Matthews Correlation Coefficient
#     kappa = cohen_kappa_score(labels, preds)  # Cohen's Kappa
    
#     # Precision-Recall curve based metrics
#     try:
#         precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probs)
#         # Area under P-R curve
#         auprc = np.trapz(precision_curve, recall_curve)
#     except:
#         auprc = 0.0
    
#     # FPR and TPR
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
#     tpr = recall  # True Positive Rate (same as recall)
    
#     # FNR and TNR
#     fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
#     tnr = specificity  # True Negative Rate (same as specificity)
    
#     # Balanced accuracy
#     balanced_acc = (tpr + tnr) / 2
    
#     # Diagnostic odds ratio
#     dor = (tp/fp)/(fn/tn) if fp > 0 and fn > 0 and tn > 0 else 0
    
#     # Threat score / Critical success index
#     ts = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
#     # F2 Score (beta=2, emphasizes recall)
#     f2 = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall) if (precision + recall) > 0 else 0
    
#     # Jaccard index
#     jaccard = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
#     return {
#         # Original metrics
#         'ACC': accuracy,
#         'AP': ap,
#         'AUC': auc,
#         'Precision': precision,
#         'Recall': recall,
#         'F1': f1,
        
#         # Extended metrics
#         'Specificity': specificity,
#         'MCC': mcc,
#         'Kappa': kappa,
#         'AUPRC': auprc,
#         'FPR': fpr,
#         'TPR': tpr,
#         'FNR': fnr,
#         'TNR': tnr,
#         'Balanced_ACC': balanced_acc,
#         'DOR': dor,
#         'Threat_Score': ts,
#         'F2_Score': f2,
#         'Jaccard': jaccard,
        
#         # Confusion matrix components
#         'TP': int(tp),
#         'TN': int(tn),
#         'FP': int(fp),
#         'FN': int(fn)
#     }

# def print_metrics_report(metrics_dict, logger):
#     """
#     Print a formatted report of the computed metrics
    
#     Args:
#         metrics_dict: Dictionary returned by compute_extended_metrics
#     """
#     logger.log("\n" + "="*60)
#     logger.log("FAKE NEWS DETECTION EVALUATION REPORT")
#     logger.log("="*60)
    
#     # Main performance metrics
#     logger.log("\nMain Performance Metrics:")
#     logger.log("-"*30)
#     logger.log(f"Accuracy     : {metrics_dict['ACC']:.4f}")
#     logger.log(f"Precision    : {metrics_dict['Precision']:.4f}")
#     logger.log(f"Recall       : {metrics_dict['Recall']:.4f}")
#     logger.log(f"F1-Score     : {metrics_dict['F1']:.4f}")
#     logger.log(f"AUC          : {metrics_dict['AUC']:.4f}")
#     logger.log(f"AP           : {metrics_dict['AP']:.4f}")
#     logger.log(f"AUPRC        : {metrics_dict['AUPRC']:.4f}")
    
#     # Advanced metrics
#     logger.log("\nAdvanced Metrics:")
#     logger.log("-"*30)
#     logger.log(f"Balanced ACC : {metrics_dict['Balanced_ACC']:.4f}")
#     logger.log(f"MCC          : {metrics_dict['MCC']:.4f}")
#     logger.log(f"Kappa        : {metrics_dict['Kappa']:.4f}")
#     logger.log(f"Specificity  : {metrics_dict['Specificity']:.4f}")
#     logger.log(f"F2-Score     : {metrics_dict['F2_Score']:.4f}")
    
#     # Error rates
#     logger.log("\nError Rates:")
#     logger.log("-"*30)
#     logger.log(f"FPR (False Positive Rate) : {metrics_dict['FPR']:.4f}")
#     logger.log(f"FNR (False Negative Rate) : {metrics_dict['FNR']:.4f}")
    
#     # Confusion Matrix
#     logger.log("\nConfusion Matrix:")
#     logger.log("-"*30)
#     logger.log(f"True Positives  (TP): {metrics_dict['TP']}")
#     logger.log(f"True Negatives  (TN): {metrics_dict['TN']}")
#     logger.log(f"False Positives (FP): {metrics_dict['FP']}")
#     logger.log(f"False Negatives (FN): {metrics_dict['FN']}")
    
#     logger.log("="*60)

# def print_metrics_report(metrics_dict, logger):
#     """
#     Print a formatted report of the computed metrics
    
#     Args:
#         metrics_dict: Dictionary returned by compute_extended_metrics
#     """
#     print("\n" + "="*60)
#     print("FAKE NEWS DETECTION EVALUATION REPORT")
#     print("="*60)
    
#     # Main performance metrics
#     print("\nMain Performance Metrics:")
#     print("-"*30)
#     print(f"Accuracy     : {metrics_dict['ACC']:.4f}")
#     print(f"Precision    : {metrics_dict['Precision']:.4f}")
#     print(f"Recall       : {metrics_dict['Recall']:.4f}")
#     print(f"F1-Score     : {metrics_dict['F1']:.4f}")
#     print(f"AUC          : {metrics_dict['AUC']:.4f}")
#     print(f"AP           : {metrics_dict['AP']:.4f}")
#     print(f"AUPRC        : {metrics_dict['AUPRC']:.4f}")
    
#     # Advanced metrics
#     print("\nAdvanced Metrics:")
#     print("-"*30)
#     print(f"Balanced ACC : {metrics_dict['Balanced_ACC']:.4f}")
#     print(f"MCC          : {metrics_dict['MCC']:.4f}")
#     print(f"Kappa        : {metrics_dict['Kappa']:.4f}")
#     print(f"Specificity  : {metrics_dict['Specificity']:.4f}")
#     print(f"F2-Score     : {metrics_dict['F2_Score']:.4f}")
    
#     # Error rates
#     print("\nError Rates:")
#     print("-"*30)
#     print(f"FPR (False Positive Rate) : {metrics_dict['FPR']:.4f}")
#     print(f"FNR (False Negative Rate) : {metrics_dict['FNR']:.4f}")
    
#     # Confusion Matrix
#     print("\nConfusion Matrix:")
#     print("-"*30)
#     print(f"True Positives  (TP): {metrics_dict['TP']}")
#     print(f"True Negatives  (TN): {metrics_dict['TN']}")
#     print(f"False Positives (FP): {metrics_dict['FP']}")
#     print(f"False Negatives (FN): {metrics_dict['FN']}")
    
#     print("="*60)
    


