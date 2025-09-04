import os
import argparse
from numpy import linalg as LA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from scipy.special import softmax


########################################################################################################################
#  Calculate Importance
########################################################################################################################

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Calculate sample-wise importance',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', 
                    help='Path to dataset')
parser.add_argument('--dynamics_path', type=str, required=True,
                    help='Folder to saved dynamics.')
parser.add_argument('--window_size', default=10, type=int,
                    help='Size of the sliding window. (for Dyn-Unc & DUAL & TDDS)')
parser.add_argument('--save_path', type=str, required=True,
                    help='Folder to save mask.')
parser.add_argument('--seed', default=42, type=int, 
                    help='manual seed')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                    help='dataset')


args = parser.parse_args()

def forgetting(rearranged, targets):
    _, pred = rearranged.max(dim=-1)
    target_ex = targets.expand(rearranged.shape[0], -1)
    correct_arr = (pred == target_ex)
    forget_events = (correct_arr[:-1] > correct_arr[1:]).sum(dim=0)
    score = forget_events 
    mask = forget_events.sort()[1]
    return score, mask

def el2n(rearranged, targets):
    one_hot = F.one_hot(targets, rearranged.shape[-1])
    rearranged = F.softmax(rearranged, dim=-1)
    l2_error = torch.norm(one_hot-rearranged, p=2, dim=-1)
    score = l2_error
    mask = torch.tensor(l2_error).sort()[1]
    return score, mask

def aum(rearranged):
    rearranged = F.softmax(rearranged, dim=2)
    for T in range(rearranged.shape[0]): # iter 200
        probs = rearranged[T]
        aum = torch.zeros(probs.shape[0])
        target_prob = probs[range(probs.size(0)), targets]
        probs[range(probs.size(0)), targets] = 0
        other_highest_prob = probs.max(dim=1)[0]
        margin = target_prob - other_highest_prob
        aum += margin
    score = aum
    mask = aum.sort()[1]
    return score, mask

def entropy(rearranged):
    prob = nn.Softmax(dim=1)(rearranged[-1])
    entropy = -1 * prob * torch.log(prob + 1e-10)
    entropy = torch.sum(entropy, dim=1).detach().cpu()
    score = entropy
    mask = entropy.sort()[1]
    return score, mask

def dynunc(preds, window_size=10, dim=0):
    windows_score = []
    for i in range(preds.size(dim) - window_size + 1):
        window = preds[i:i+window_size, :] 
        win_std = window.std(dim=0) * 10
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

def rearrange(args, probs, indexes):
    probs_window_rere = []
    # Reorganize probabilities according to indexes
    for i in range(probs.shape[0]):
        probs_window_re = torch.zeros_like(torch.tensor(probs[i]))
        probs_re = probs_window_re.index_add(0, torch.tensor(indexes[i], dtype=int), torch.tensor(probs[i]))
        probs_window_rere.append(probs_re)

    rearranged = torch.stack(probs_window_rere)
    target_probs = F.softmax(rearranged, dim=2)
    
    if args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(args.data_path, download=True, train=True)
    elif args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(args.data_path, download=True, train=True)
        
    targets = torch.tensor(train_data.targets)
    targets_expanded = targets.unsqueeze(0).unsqueeze(2).expand(probs.shape[0], -1, 1)
    target_probs = torch.gather(target_probs, 2, targets_expanded).squeeze()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    np.save(f"{args.save_path}/rearranged.npy", rearranged)
    np.save(f"{args.save_path}/target_probs.npy", target_probs)
    
    return targets, targets_expanded, target_probs, rearranged

    
if __name__ == '__main__':
    dynamics_path = args.dynamics_path
    probs = []
    losses = []
    indexes = []
    
    for i in range(200):
        probs.append(np.load(dynamics_path + str(i) + '_Output.npy'))
        losses.append(np.load(dynamics_path + str(i) + '_Loss.npy'))
        indexes.append(np.load(dynamics_path + str(i) + '_Index.npy'))

    probs = np.array(probs)
    losses = np.array(losses)
    indexes = np.array(indexes)

    print("Rearrange target probability")
    targets, targets_expanded, target_probs, rearranged = rearrange(args, probs, indexes)
    
    # DUAL
    print("DUAL score processing (T=30)")
    score, mask = dual(target_probs[:30])
    np.save(os.path.join(args.save_path, 'dual_score_T30.npy'), score)
    np.save(os.path.join(args.save_path, 'dual_mask_T30.npy'), mask)
    
    # DYNUNC
    print("DYN-UNC score processing")
    score, mask = dynunc(target_probs)
    np.save(os.path.join(args.save_path, 'dynunc_score.npy'), score)
    np.save(os.path.join(args.save_path, 'dynunc_mask.npy'), mask)
    
    # EL2N
    print("EL2N score processing")
    score, mask = el2n(rearranged[20], targets)
    np.save(os.path.join(args.save_path, 'el2n_score.npy'), score)
    np.save(os.path.join(args.save_path, 'el2n_mask.npy'), mask)
    
    # AUM
    print("AUM score processing")
    score, mask = aum(rearranged)
    np.save(os.path.join(args.save_path, 'aum_score.npy'), score)
    np.save(os.path.join(args.save_path, 'aum_mask.npy'), mask)
    
    # TDDS
    print("TDDS score processing")
    T, J = 90, 10
    score, mask = tdds(T, J, rearranged)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_score.npy"), score)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_mask.npy"), mask)
    
    T, J = 70, 10
    score, mask = tdds(T, J, rearranged)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_score.npy"), score)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_mask.npy"), mask)
    
    T, J = 60, 20
    score, mask = tdds(T, J, rearranged)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_score.npy"), score)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_mask.npy"), mask)
    
    T, J = 20, 10
    score, mask = tdds(T, J, rearranged)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_score.npy"), score)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_mask.npy"), mask)
    
    T, J = 10, 5
    score, mask = tdds(T, J, rearranged)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_score.npy"), score)
    np.save(os.path.join(args.save_path, f"tdds_{T}_{J}_mask.npy"), mask)
    
    # ENTROPY
    score, mask = entropy(rearranged)
    np.save(os.path.join(args.save_path, f"entropy_score.npy"), score)
    np.save(os.path.join(args.save_path, f"entropy_mask.npy"), mask)
    
    # FORGETTING
    score, mask = forgetting(rearranged, targets)
    np.save(os.path.join(args.save_path, f"forgetting_score.npy"), score)
    np.save(os.path.join(args.save_path, f"forgetting_mask.npy"), mask)
