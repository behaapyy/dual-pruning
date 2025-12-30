import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()

######################### Data Setting #########################
parser.add_argument('--td-path', type=str, default='', help='The dir path of the training dynamics saved')
parser.add_argument('--label-path', type=str, default='', help='The path of gt labels')
parser.add_argument('--save-path', type=str, default='', help='The save path of score and mask files')

args = parser.parse_args()

def dynunc(preds, window_size=10, dim=0):
    windows_score = []
    for i in range(preds.size(dim) - window_size + 1):
        window = preds[i:i+window_size, :] 
        win_std = window.std(dim=0) * 10
        win_mean = window.mean(dim=0)
        windows_score.append(win_std)
    score = torch.stack(windows_score).mean(dim=0)
    mask = np.argsort(score)
    return score, mask

def tdds(T, J, rearranged):
    # Calculate TDDS
    k = 0
    moving_averages = []
    # Iterate through the trajectory
    while k < T - J + 1:
        probs_window_kd = []
        # Calculate KL divergence in one window
        for j in range(J - 1):
            log = torch.log(rearranged[j + 1] + 1e-8) - torch.log(rearranged[j] + 1e-8)
            kd = torch.abs(torch.multiply(rearranged[j + 1], log.nan_to_num(0))).sum(axis=1)
            probs_window_kd.append(kd)
        window_average = torch.stack(probs_window_kd).mean(dim=0)
        
        window_diff = []
        for ii in range(J - 1):
            window_diff.append((probs_window_kd[ii] - window_average))
        window_diff_norm = LA.norm(torch.stack(window_diff), axis=0) 
        moving_averages.append(window_diff_norm * 0.9 * (1 - 0.9) ** (T - J - k))
        k += 1
        
    moving_averages_sum = np.squeeze(sum(np.array(moving_averages), 0))
    mask = moving_averages_sum.argsort()
    score = moving_averages_sum
    return score, mask

def dual(preds, window_size=10, dim=0):
    windows_score = []
    for i in range(preds.size(dim) - window_size + 1):
        window = preds[i:i+window_size, :] 
        win_std = window.std(dim=0) * 10
        win_mean = window.mean(dim=0)
        windows_score.append((win_std * (1-win_mean)))
    score = torch.stack(windows_score).mean(dim=0)
    mask = np.argsort(score)
    return score, mask


for i, filename in enumerate(os.listdir(args.td_path)):
    td_path = os.path.join(args.td_path, filename)
    with open(td_path, 'rb') as f:
        td_data = pickle.load(f)
    
    grouped_data = defaultdict(lambda: {'idx': [], 'output': []})
    for entry in td_data['training_dynamics']:
        epoch = entry['epoch']
        grouped_data[epoch]['idx'].append(entry['idx'])
        grouped_data[epoch]['output'].append(entry['output'])
    
    total_result = {}
    for epoch, tensors in grouped_data.items():
        total_result[epoch] = {
            'idx': torch.cat(tensors['idx']),
            'output': torch.cat(tensors['output'])
        }

idxs = []
outputs = []

for epoch in total_result.keys():  
    idx = total_result[epoch]['idx']
    output = total_result[epoch]['output']
    idxs.append(idx)
    outputs.append(output)

idxs = torch.stack(idxs, dim=0)
outputs = torch.stack(outputs, dim=0)

probs_rearranged = []
for i in range(idxs.shape[0]): # epoch
    probs_re = torch.zeros_like(torch.tensor(outputs[i]))
    probs_re = probs_re.index_add(0, idxs[i].type(torch.int64), torch.tensor(outputs[i]))
    probs_rearranged.append(probs_re)
rearranged = torch.stack(probs_rearranged)
rearranged = F.softmax(rearranged, dim=-1)

labels = np.load(args.label_path)
labels_t = torch.from_numpy(labels).long().to(rearranged.device)

labels_expanded = labels_t.view(1, -1, 1).expand(rearranged.size(0), -1, 1)

target_probs = torch.gather(rearranged, dim=2, index=labels_expanded).squeeze(-1)


score, mask = dynunc(target_probs, window_size=10, dim=0)
np.save(os.path.join(args.save_path, 'dynunc_score'), score)
np.save(os.path.join(args.save_path, 'dynunc_mask'), mask)


score, mask = dual(target_probs)
np.save(os.path.join(args.save_path, 'dual_score'), score)
np.save(os.path.join(args.save_path, 'dual_mask'), mask)

score, mask = tdds(70, 10, rearranged)
np.save(os.path.join(args.save_path, 'tdds_score'), score)
np.save(os.path.join(args.save_path, 'tdds_mask'), mask)
