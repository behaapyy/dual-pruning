import os, sys, shutil, time, random
import glob
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from models import resnet
import numpy as np
import math
import wandb
from data_subset import load_cifar100_sub, load_cifar10_sub
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

########################################################################################################################
#  Training Subset
########################################################################################################################

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',choices=['cifar10', 'cifar100'],
                    help='Choose between Cifar10 and 100.')
parser.add_argument('--arch', type=str, default='resnet18')

# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./ckpt', help='Folder to save checkpoints and log.')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--evaluate', dest='evaluate', action='store_true',default= False, help='evaluate model on validation set')

# Pruning
parser.add_argument('--cutoff_rate', default=0.1, type=float)
parser.add_argument('--subset_rate', default=0.3, type=float, help='subset rate')

# for d2
parser.add_argument('--budget_mode', default='uniform', type=str, help='d2 inference')
parser.add_argument('--n_neighbor', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--stratas', default=50)
parser.add_argument('--graph-mode', default='sum')
parser.add_argument('--graph-sampling-mode', default='weighted')
parser.add_argument('--mis-ratio', default=0.2, type=float)

# Acceleration
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

# scoring params
parser.add_argument('--src-folder', help='train dynamics source')

# Coreset Method
parser.add_argument('--target-probs-path', default='./generated/cifar10/42/target_probs_win_10_ep200.npy', type=str, help='for dual + beta')
parser.add_argument('--score-path', default='./generated/cifar10/42/dual_mask_T30.npy', type=str)
parser.add_argument('--mask-path', default='./generated/cifar10/42/dual_mask_T30.npy', type=str)
parser.add_argument('--c_d', type=float, default=4, help='d_c for beta sampling')
parser.add_argument('--key_T', type=int, default=30, help='score computation epoch')
parser.add_argument('--key_J', type=int, default=10, help='sliding window size')
parser.add_argument('--sample', type=str, default=None, help='sampling method: random, stratified, beta')
parser.add_argument('--method', type=str, default='dual', help='methodology name')

args = parser.parse_args()
args.use_cuda = True
args.device = f'cuda:{args.gpu}'
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

wandb.init(project=f"{args.arch}_{args.dataset}_{args.arch}", 
           name = f"{args.method}_{args.sample}_{args.subset_rate}_bsz{args.batch_size}",
           config=args)

def main(): # 
    # Init logger
    print(args.save_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        
    log = open(os.path.join(args.save_path, f'{args.sample}_log.txt'), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Dataset: {}".format(args.dataset), log)
    print_log("Data Path: {}".format(args.data_path), log)
    print_log("Network: {}".format(args.arch), log)
    print_log("Batchsize: {}".format(args.batch_size), log)
    print_log("Learning Rate: {}".format(args.learning_rate), log)
    print_log("Momentum: {}".format(args.momentum), log)
    print_log("Weight Decay: {}".format(args.decay), log)

    target_probs = np.load(args.target_probs_path)[:1]
    score = np.load(args.score_path)
    mask = np.load(args.mask_path)
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.num_samples = 50000 * args.subset_rate
        args.num_iter = math.ceil(args.num_samples/args.batch_size)
        train_loader, test_loader = load_cifar10_sub(args, target_probs, score, mask)
        
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.num_samples = 50000 * args.subset_rate
        args.num_iter = math.ceil(args.num_samples/args.batch_size)
        train_loader, test_loader = load_cifar100_sub(args, target_probs, score, mask)
    else:
        raise NotImplementedError("Unsupported dataset type")
    
    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    if args.arch in ['resnet18', 'resnet50']:
        net = resnet.__dict__[args.arch](num_class = args.num_classes)
    elif args.arch == 'vgg':
        from models import vgg
        net = vgg.VGG('VGG16', num_class=args.num_classes)
    
    net = net.to(args.device)
    print_log("=> network :\n {}".format(net), log)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max = args.epochs * args.num_iter)

    recorder = RecorderMeter(args.epochs)
    # evaluation
    if args.evaluate:
        time1 = time.time()
        validate(test_loader, args, net, criterion, log) #
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.epochs):
        # current_learning_rate = scheduler.get_last_lr()[0]
        current_learning_rate = scheduler.get_lr()[0]
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        ###
        # print(len(train_loader))
        train_acc, train_los = train(train_loader, args, net, criterion, optimizer, scheduler, epoch, log)

        # evaluate on validation set
        val_acc, val_los = validate(test_loader, args, net, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_los,
            "val_accuracy": val_acc,
            "val_loss": val_los,
            "learning_rate": scheduler.get_last_lr()[0],
            # "learning_rate": scheduler.get_lr()[0],
            "best_acc": recorder.max_accuracy(False)
        })
        
        if args.save:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net,
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_path, f'{args.dataset}_{args.subset_rate}_{args.manualSeed}_{args.sample}_checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
    log.close()
    wandb.finish()

# train function (forward, backward, update)
def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    
    for t, (input, target) in enumerate(train_loader):
        if args.use_cuda:
            y = target.to(args.device)
            x = input.to(args.device)
        
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(y)
        # compute output
        output = model(input_var)
        n = len(y)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, t, args.batch_size, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(test_loader, args, model, criterion, log): 
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if args.use_cuda:
                y = target.to(args.device)
                x = input.to(args.device)

            # compute output
            output = model(x)
            loss = criterion(output, y)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
            losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))

        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                    error1=100 - top1.avg),
                log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, f'best.pth.tar')
        shutil.copyfile(filename, bestname)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main() 
