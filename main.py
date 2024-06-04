'''Train Cifar10 or Caltech101 or Imbalanced Cifar10 with PyTorch. PT4AL'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import ResNet18, ResNet_Caltech101
from supervised_loader import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch Cifar10 and Caltech101 Training')
parser.add_argument('--dataset', default='Cifar10', type=str, help='dataset (Cifar10 or Imbalanced_Cifar10 or Caltech101)')
parser.add_argument('--sampling', default='confidence', type=str, help='sampling method (confidence, entropy)')
parser.add_argument('--sample_ratio', default=0.2, type=float, help='sampling ratio')
parser.add_argument('--task', default='rotation', type=str, help='pretext task (rotation, colorization)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data preparation
print('==> Preparing data..')
if args.dataset == 'Cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = Loader_Cifar10(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    net = ResNet18()
    
elif args.dataset == 'Imbalanced_Cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = Loader3_Cifar10(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    net = ResNet18()
    
    
else:
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    testset = Loader_Caltech101(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    net = ResNet_Caltech101()
    num_classes = testset.num_classes
    net.linear = nn.Linear(net.linear.in_features, num_classes)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing
def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{args.task}'):
            os.mkdir(f'checkpoint_{args.dataset}_{args.task}')
        torch.save(state, f'./checkpoint_{args.dataset}_{args.task}/main_{cycle}.pth')
        best_acc = acc

# confidence sampling (pseudo labeling)
def get_plabels_confidence(net, samples, cycle, sample_ratio=0.2):
    if args.dataset == 'Cifar10':
        sub5k = Loader2_Cifar10(is_train=False, transform=transform_test, path_list=samples)
    elif args.dataset == 'Imbalanced_Cifar10':
        sub5k = Loader4_Cifar10(is_train=False, transform=transform_test, path_list=samples)
    else:
        samples = [p.strip().replace('Caltech101/DATA/', '') for p in samples]
        sub5k = Loader2_Caltech101(is_train=False, transform=transform_test, path_list=samples)


    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()].cpu())
            progress_bar(idx, len(ploader))

    top1_scores = np.array(top1_scores)  
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    
    num_samples = int(len(samples) * sample_ratio)
    return samples[idx[:num_samples]]

# entropy sampling
def get_plabels_entropy(net, samples, cycle, sample_ratio=0.2):
    if args.dataset == 'Cifar10':
        sub5k = Loader2_Cifar10(is_train=False, transform=transform_test, path_list=samples)
    elif args.dataset == 'Imbalanced_Cifar10':
        sub5k = Loader4_Cifar10(is_train=False, transform=transform_test, path_list=samples)
    else:
        samples = [p.strip().replace('Caltech101/DATA/', '') for p in samples]
        sub5k = Loader2_Caltech101(is_train=False, transform=transform_test, path_list=samples)

    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=0)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    top1_scores = np.array(top1_scores)
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    
    num_samples = int(len(samples) * sample_ratio)
    return samples[idx[-num_samples:]]

if __name__ == '__main__':
    labeled = []
        
    CYCLES = 10
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        with open(f'./loss_{args.dataset}_{args.task}/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
            
        if cycle > 0:
            print('>> Getting previous checkpoint')
            checkpoint = torch.load(f'./checkpoint_{args.dataset}_{args.task}/main_{cycle-1}.pth')
            net.load_state_dict(checkpoint['net'])

            if args.sampling == 'confidence':
                sample1k = get_plabels_confidence(net, samples, cycle)
            elif args.sampling == 'entropy':
                sample1k = get_plabels_entropy(net, samples, cycle, sample_ratio=args.sample_ratio)

        else:
            samples = np.array(samples)
            num_samples = int(len(samples) * args.sample_ratio)
            sample1k = samples[:num_samples]

        labeled.extend(sample1k)
        print(f'>> Labeled length: {len(labeled)}')

        if args.dataset == 'Cifar10':
            trainset = Loader2_Cifar10(is_train=True, transform=transform_train, path_list=labeled)
        elif args.dataset == 'Imbalanced_Cifar10':
            trainset = Loader4_Cifar10(is_train=True, transform=transform_train, path_list=labeled)
        else:
            trainset = Loader2_Caltech101(is_train=True, transform=transform_train, path_list=labeled)
    

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        for epoch in range(200):
            train(net, criterion, optimizer, epoch, trainloader)
            test(net, criterion, epoch, cycle)
            scheduler.step()
        with open(f'./main_best_{args.dataset}_{args.task}.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')
