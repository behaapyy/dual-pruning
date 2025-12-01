import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
from core.data import CoresetSelection, IndexDataset, CIFARDataset, ImageNetDataset
import numpy as np
from numpy import linalg as LA

parser = argparse.ArgumentParser()

######################### Data Setting #########################
parser.add_argument('--data-dir', type=str, default='/storage/dataset/imagenet',
                    help='The dir path of the imagenet')
parser.add_argument('--base-dir', type=str)
parser.add_argument('--task-name', type=str)
parser.add_argument('--data-score-path', type=str)

args = parser.parse_args()

def EL2N(td_log, data_importance, max_epoch):
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        index = td_log['idx'].type(torch.long)

        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))
        data_importance['el2n'][index] += el2n_score


    for i, item in enumerate(td_log):
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)

def training_dynamics_metrics(td_log, data_importance):
    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        record_training_dynamics(item)

#Load all data
data_dir = args.data_dir
trainset = ImageNetDataset.get_ImageNet_train(os.path.join(data_dir, 'train'))
trainset = IndexDataset(trainset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=800, shuffle=False, pin_memory=True, num_workers=16)

# Load all targets into array
targets = []
print(f'Load label info from datasets...')
print(f'Total batch: {len(trainloader)}')
for batch_idx, (idx, (_, y)) in enumerate(trainloader):
    targets += list(y.numpy())
    if batch_idx % 50 == 0:
        print(batch_idx)

print(len(targets))
data_importance = {}
targets = torch.tensor(targets)
data_size = targets.shape[0]
num_classes = 1000

data_importance['targets'] = targets.type(torch.int32)
data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

for i in range(1,11):
    td_path = f"{args.base_dir}/{args.task_name}/training-dynamics/td-{args.task_name}-epoch-{i}.pickle"
    print(td_path)
    with open(td_path, 'rb') as f:
         td_data = pickle.load(f)
    EL2N(td_data['training_dynamics'], data_importance, max_epoch=11)

total_index, total_target_probs = [], []
for i in range(1,61):
    td_path = f"{args.base_dir}/{args.task_name}/training-dynamics/td-{args.task_name}-epoch-{i}.pickle"
    print(td_path)
    with open(td_path, 'rb') as f:
         td_data = pickle.load(f)
    epoch_index, epoch_target_probs = training_dynamics_metrics(td_data['training_dynamics'], data_importance)
    total_index.append(epoch_index)
    total_target_probs.append(epoch_target_probs)

np.save(os.path.dirname(args.data_score_path)+'/index.npy', torch.stack(total_index))
np.save(os.path.dirname(args.data_score_path)+'/target_probs.npy', torch.stack(total_target_probs))

print(f'Saving data score at {args.data_score_path}')
with open(args.data_score_path, 'wb') as handle:
    pickle.dump(args.data_importance, handle)
    

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
