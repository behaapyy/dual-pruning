import os
import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy import stats
from scipy.stats import beta
from scipy.stats import betabinom

# from scipy.special import softmax
import math
from dynm_utils import beta_2d, plot_pdf
from d2_sampling import GraphDensitySampler
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, data_mask, score, target_probs):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z
    
    if args.sample == 'beta_2d':
        # data_mean = target_probs[:30].mean(axis=0)
        # subset_mask = beta_2d(1-args.subset_rate, args.data_simplicity, score, data_mask, data_mean)[0]
        # logging figure
        sns.set_style('whitegrid')
        # args.subset_rate = pruning rate
        remain_id, pred_std, pred_mean, px, py, joint_pxy = beta_2d(1-args.subset_rate, args.data_simplicity, target_probs, score, data_mask)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        plot_pdf(axs[0], pred_std, pred_mean, px, py)
        axs[0].legend()
        axs[0].set_title(f"simplicity: {args.data_simplicity}, {int((1-args.subset_rate)*100)}% pruning")

        axs[1].scatter(pred_std, pred_mean, c=joint_pxy, s=0.5, cmap='plasma', alpha=0.2)
        axs[1].set_title('sampling probability')
        
        axs[2].scatter(pred_std[remain_id], pred_mean[remain_id], c=score[np.argsort(data_mask)][remain_id],  s=0.5, alpha=0.3, label='score', cmap='viridis')
        axs[2].set_title("selected")
        axs[2].set_xlim(0, 0.45)
        axs[2].set_ylim(0, 1)
        
        wandb.log({"pruning visualization": wandb.Image(fig)})
        subset_mask = remain_id
    
    elif args.sample == 'random':
        n = int(args.subset_rate * len(data_mask))
        subset_mask = np.random.choice(range(len(train_data)), size=n, replace=False)
    
    elif args.sample == 'stratified':
        n = int(args.subset_rate * len(data_mask))
        selected_index, _ = stratified_sampling(torch.tensor(score), coreset_num=n)
        subset_mask = data_mask[selected_index]
    
    elif args.sample == 'ccs':
        n = int(args.subset_rate * len(data_mask))
        mis_n = int(args.mis_ratio * len(data_mask))
        
        selected_index, _ = stratified_sampling(torch.tensor(score[mis_n:]), coreset_num=n)
        subset_mask = data_mask[selected_index]
        
        score = np.load('')
        mask = np.load('')
        easy_index = mask[mis_n:]
        
        selected_index, _ = stratified_sampling(torch.tensor(score[mis_n:]), coreset_num=n)
        subset_mask = easy_index[selected_index]
        
    
    elif args.sample == 'moderate':
        subset_mask = data_mask
    
    else:
        subset_mask = data_mask[-int(args.subset_rate * len(data_mask)):] # ascending
        pred_std, pred_mean = target_probs.std(axis=0), target_probs.mean(axis=0)
        sns.set_style('whitegrid')
        fig, axs = plt.subplots()
        axs.scatter(pred_std[subset_mask], pred_mean[subset_mask], c=score[np.argsort(data_mask)][subset_mask],  s=0.5, alpha=0.3, label='score', cmap='viridis')
        axs.set_title("selected")
        axs.set_xlim(0, 0.45)
        axs.set_ylim(0, 1)
        wandb.log({"pruning visualization": wandb.Image(fig)})
    
    # subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar100_sub(args, data_mask, score, target_probs=None):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z
    
    ###        
    if args.sample == 'random':
        n = int(args.subset_rate * len(data_mask))
        subset_mask = np.random.choice(range(len(train_data)), size=n, replace=False)
        
    elif args.sample == 'stratified':
        n = int(args.subset_rate * len(data_mask))
        selected_index, _ = stratified_sampling(torch.tensor(score), coreset_num=n)
        subset_mask = data_mask[selected_index]

    elif args.sample == 'beta':
        # data_mean = target_probs[:30].mean(axis=0)
        sns.set_style('whitegrid')
        remain_id, pred_std, pred_mean, px, py, joint_pxy = beta_2d(1-args.subset_rate, args.data_simplicity, target_probs, score, data_mask)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        plot_pdf(axs[0], pred_std, pred_mean, px, py)
        axs[0].legend()
        axs[0].set_title(f"simplicity: {args.data_simplicity}, {int((1-args.subset_rate)*100)}% pruning")

        axs[1].scatter(pred_std, pred_mean, c=joint_pxy, s=0.5, cmap='plasma', alpha=0.2)
        axs[1].set_title('sampling probability')
        
        axs[2].scatter(pred_std[remain_id], pred_mean[remain_id], c=score[np.argsort(data_mask)][remain_id],  s=0.5, alpha=0.3, label='score', cmap='viridis')
        axs[2].set_title("selected")
        axs[2].set_xlim(0, 0.45)
        axs[2].set_ylim(0, 1)
        
        wandb.log({"pruning visualization": wandb.Image(fig)})
        subset_mask = remain_id
        
    elif args.sample == 'ccs':
        n = int(args.subset_rate * len(data_mask))
        mis_n = int(args.mis_ratio * len(data_mask))
        
        score = np.load('')
        mask = np.load('')
        easy_index = mask[mis_n:]
        
        selected_index, _ = stratified_sampling(torch.tensor(score[mis_n:]), coreset_num=n)
        subset_mask = easy_index[selected_index]
        
    else: # get highest score samples
        subset_n = int(args.subset_rate * len(data_mask))
        subset_mask = data_mask[-subset_n:]
    
    data_set = torch.utils.data.Subset(train_data, subset_mask)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


def stratified_sampling(score, coreset_num):
        stratas = 50
        print('Using stratified sampling...')

        total_num = coreset_num
        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num

            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(score.shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]
        
        return selected_index, None