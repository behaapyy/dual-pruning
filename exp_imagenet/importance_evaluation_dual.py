import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from numpy import linalg as LA
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

parser = argparse.ArgumentParser()

######################### Data Setting #########################
parser.add_argument('--td-path', type=str, default='', help='The dir path of the training dynamics saved')
parser.add_argument('--task-name', type=str)

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
    td_path = os.path.join(td_dir, filename)
    with open(td_path, 'rb') as f:
        td_data = pickle.load(f)
    
    grouped_data = defaultdict(lambda: {'idx': [], 'output': []})
    for entry in td_data['training_dynamics']:
        epoch = entry['epoch']
        grouped_data[epoch]['idx'].append(entry['idx'])
        grouped_data[epoch]['output'].append(entry['output'])
    
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
    print(outputs)

idxs = torch.stack(idxs, dim=0)
outputs = torch.stack(outputs, dim=0)

probs_rearranged = []
for i in range(idxs.shape[0]): # epoch
    probs_re = torch.zeros_like(torch.tensor(outputs[i]))
    probs_re = probs_re.index_add(0, idxs[i].type(torch.int64), torch.tensor(outputs[i]))
    probs_rearranged.append(probs_re)
rearranged = torch.stack(probs_rearranged)

score, mask = dynunc(rearranged, window_size=10, dim=0)
np.save('', score) # please fill the path for saving score and mask
np.save('', mask)

score, mask = dual(rearranged)
np.save('', score)
np.save('', mask)

score, mask = tdds(70, 10, rearranged)
np.save('', score)
np.save('', mask)
