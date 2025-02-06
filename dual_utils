import torch
from scipy.stats import beta
import numpy as np


def beta_sampling(prune_rate, c_d, target_probs, score, mask):
    data_length = target_probs.shape[-1]
    target_probs = torch.tensor(target_probs)
    pred_std = target_probs.std(axis=0)
    pred_mean = target_probs.mean(axis=0)

    subset_n = int((1-prune_rate) * data_length)
    anchor_mean = pred_mean[mask[-10:]].mean()
    y_b = 15 * (1-anchor_mean) * (1 -prune_rate**(2+c_d))
    y_a = 16 - y_b
    
    pdf_y = beta.pdf(pred_mean, y_a, y_b)
    joint_p = pdf_y * score
    remain_id = np.random.choice(data_length, p=joint_p/joint_p.sum(), size=subset_n, replace=False)

    return remain_id, pred_std, pred_mean, pdf_y[np.argsort(pred_mean)], joint_p