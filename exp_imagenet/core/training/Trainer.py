import torch
import torch.nn.functional as F
from datetime import datetime
from ..utils import accuracy
import wandb
import numpy as np
# import habana_frameworks.torch.core as htcore

class Trainer(object):
    """
    Helper class for training.
    """

    def __init__(self):
        pass

    """
    Dataset need to be an index dataset.
    Set remaining_iterations to -1 to ignore this argument.
    """
    def train(self, epoch, remaining_iterations, model, dataloader, optimizer, criterion, scheduler, device, TD_logger=None, log_interval=None, printlog=False):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()
        if printlog:
            print('*' * 26)
        for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # s = score[np.isin(mask, idx)]
            # loss *= s.mean()
            
            loss.backward()
            # htcore.mark_step()
            optimizer.step()
            # htcore.mark_step()
            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

            if TD_logger:
                log_tuple = {
                'epoch': epoch,
                'iteration': batch_idx,
                'idx': idx.type(torch.long).clone(),
                'output': F.log_softmax(outputs, dim=1).detach().cpu().type(torch.half)
                }
                TD_logger.log_tuple(log_tuple)

            if printlog and log_interval and batch_idx % log_interval == 0:
                print(f"{batch_idx}/{len(dataloader)}")
                print(f'>> batch_idx [{batch_idx}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')

            remaining_iterations -= 1
            if remaining_iterations == 0:
                if printlog: print("Exit early in epoch training.")
                break
        
        wandb.log({
            'epoch': epoch,
            "train_acc":correct/total * 100,
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            })
        
        if printlog:
            print(f'>> Epoch [{epoch}]: Loss: {train_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'>> Epoch [{epoch}]: Training Accuracy: {correct/total * 100:.2f}')
            print(f'>> Epoch [{epoch}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
    
    
    
    def test(self, model, dataloader, criterion, device, log_interval=None,  printlog=False, topk = 1):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()

        if printlog: print('======= Testing... =======')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.mean().item()
                _, predicted = outputs.max(1)
                total += targets.shape[0]
                batch_acc, batch_correct = accuracy(outputs, targets, topk=topk)
                correct += batch_correct
                # correct += predicted.eq(targets).sum().item()

                # if printlog and log_interval and batch_idx % log_interval == 0:
                #     print(batch_idx)

        if printlog:
            print(f'Loss: {test_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'Test Accuracy: {correct/total * 100:.2f}')

        print(f'>> Test time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
        return test_loss, correct / total
    

    def test_linear(self, backbone, model, dataloader, criterion, device, log_interval=None,  printlog=False, topk = 1):
        backbone.eval()
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()

        if printlog: print('======= Testing... =======')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(backbone(inputs))
                loss = criterion(outputs, targets)

                test_loss += loss.mean().item()
                _, predicted = outputs.max(1)
                total += targets.shape[0]
                batch_acc, batch_correct = accuracy(outputs, targets, topk=topk)
                correct += batch_correct
                # correct += predicted.eq(targets).sum().item()

                # if printlog and log_interval and batch_idx % log_interval == 0:
                #     print(batch_idx)

        if printlog:
            print(f'Loss: {test_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'Test Accuracy: {correct/total * 100:.2f}')

        print(f'>> Test time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
        return test_loss, correct / total