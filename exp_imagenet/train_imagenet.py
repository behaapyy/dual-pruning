import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import glob
import argparse
import pickle
import numpy as np
from datetime import datetime

from torchvision import models

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import CoresetSelection, IndexDataset, CIFARDataset, ImageNetDataset
from core.utils import print_training_info, StdRedirect
import sys 
# sys.path.append('../')
# from dual_utils import beta_sampling
import matplotlib.pyplot as plt
import seaborn as sns
import wandb 
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
    y_b = 15 * (1-anchor_mean) * (1 -prune_rate**c_d)
    y_a = 16 - y_b
    
    pdf_y = beta.pdf(pred_mean, y_a, y_b)
    joint_p = pdf_y * score
    remain_id = np.random.choice(data_length, p=joint_p/joint_p.sum(), size=subset_n, replace=False)

    return remain_id, pred_std, pred_mean, pdf_y[np.argsort(pred_mean)], joint_p

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch IMageNet Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, default=300000, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet34', choices=['resnet50', 'resnet34'])
parser.add_argument('--scheduler', type=str, default='cosine', choices=['default', 'short', 'cosine', 'short-400k'])

parser.add_argument('--ignore-td', action='store_true', default=False)

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='/storage/dataset/imagenet',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str, default='./data-model/imagenet/',
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str)
parser.add_argument('--data-score-path', type=str, default='./imagenet/mask.pt')
parser.add_argument('--coreset-key', type=str)
parser.add_argument('--data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0,
                    help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)
parser.add_argument('--score-npy-path', type=str)
parser.add_argument('--mask-npy-path', type=str)


#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str)
parser.add_argument('--mis-data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float,
                    help="Ratio for the coreset, not the whole dataset.")

parser.add_argument('--strata', type=int, default=50)

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='4,5',
                    help='The ID of GPU.')
parser.add_argument('--local_rank', type=str)

### for DUAL
parser.add_argument('--d_c', type=float, help='d_c for dual + beta', default=11)
parser.add_argument('--probs-path', type=str, help='rearranged probability path', default=11)


args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"

args.task_name = f'{args.coreset_mode}_{args.coreset_key}_{args.coreset_ratio}'
wandb.init(project=f"DataPruning_ImageNet_{args.network}_{args.batch_size}_{args.coreset_mode}",
           name = args.task_name,
           config=args)
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
td_dir = os.path.join(task_dir, 'training-dynamics')
os.makedirs(td_dir, exist_ok=True)

last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')


######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')
print(f'Training log path: {td_dir}')

GPUID = args.gpuid
print(GPUID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = args.data_dir
trainset = ImageNetDataset.get_ImageNet_train(os.path.join(data_dir, 'train'))
print(trainset)
######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)

if args.coreset:
    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset), num=args.coreset_ratio * len(trainset))
    else:
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)

        print(f'Strata: {args.strata}')
        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'dual':
        mask = np.load(args.mask_npy_path)
        n = int(coreset_ratio * total_num)
        coreset_index = mask[-n:]
        
    if args.coreset_mode == 'dual_beta':
        score = np.load(args.score_npy_path)
        mask = np.load(args.mask_npy_path)
        with open(args.probs_path, 'rb') as f:
            target_probs = pickle.load(f)
        remain_id, pred_std, pred_mean, px, py, joint_pxy = beta_sampling(1-args.coreset_ratio, args.c_d, target_probs, score, mask)
        coreset_index = remain_id
    
    if args.coreset_mode == 'dynunc':
        mask = np.load(args.mask_npy_path)
        n = int(coreset_ratio * total_num)
        coreset_index = mask[-n:]

    if args.coreset_mode == 'tdds':
        mask = np.load(args.mask_npy_path)
        n = int(coreset_ratio * total_num)
        coreset_index = mask[-n:]
        
    trainset = torch.utils.data.Subset(trainset, coreset_index)
    print(len(trainset))
######################### Coreset Selection end #########################

trainset = IndexDataset(trainset)
print(len(trainset))

testset = ImageNetDataset.get_ImageNet_test(os.path.join(data_dir, 'val'))
print(len(testset))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=48)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size * 2, shuffle=True, pin_memory=True, num_workers=48)

iterations_per_epoch = len(trainloader)
print(iterations_per_epoch)

if args.network == 'resnet34':
    print('Using resnet34.')
    model = torchvision.models.resnet34(pretrained=False, progress=True)
if args.network == 'resnet50':
    print('Using resnet50.')
    model = torchvision.models.resnet50(pretrained=False, progress=True)

model=torch.nn.parallel.DataParallel(model).to(device)
# import pdb; pdb.set_trace()
model = model.to(device)
# model=model.cuda()

if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

epoch_per_testing = max(args.iterations_per_testing // iterations_per_epoch, 1)

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print(f'Using scheduler: {args.scheduler}!')
if args.scheduler == 'default':
    scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,100], gamma=0.1)
    scheduler_iteration = None
elif args.scheduler == 'short': # total epoch 70
    # scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,65], gamma=0.1)
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80000, 160000, 240000, 270000], gamma=0.1)
elif args.scheduler == 'short-400k': # total epoch 70
    # scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,65], gamma=0.1)
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 200000, 300000, 350000], gamma=0.1)
elif args.scheduler == 'cosine': # total epoch 70
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)

trainer = Trainer()

best_acc = 0
best_epoch = -1

current_epoch = 0
while num_of_iterations > 0:
    if args.ignore_td:
        TD_logger = None
        print('Ignore training dynamics info.')
    else:
        TD_logger = TrainingDynamicsLogger()
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler_iteration, device, TD_logger=TD_logger, log_interval=1000, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=5)
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=1)
        wandb.log({
                "test_loss": test_loss,
                "test_acc": test_acc
                })
        
        if test_acc > best_acc:
            print('Updating best ckpt.')
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
            wandb.log({
                "best_accuracy": best_acc
                })

    current_epoch += 1

    if scheduler_epoch:
        scheduler_epoch.step()
        print(f'Current learing rate: {scheduler_epoch.get_last_lr()}.')
    else:
        print(f'Current learing rate: {scheduler_iteration.get_last_lr()}.')

    if not args.ignore_td:
        td_path = os.path.join(td_dir, f'td-{args.task_name}-epoch-{current_epoch}.pickle')
        print(f'Saving training dynamics at {td_path}')
        TD_logger.save_training_dynamics(td_path, data_name='imagenet')

print('Last ckpt evaluation.')
# test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=5)
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=1)

print('done')
print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)

state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)