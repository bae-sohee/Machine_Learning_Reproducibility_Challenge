'''Train Cifar10 or Imbalanced_Cifar10 or Caltech101 with PyTorch. random'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import ResNet18, ResNet_Caltech101
from supervised_loader import Loader_Cifar10, Loader_Caltech101, Loader3_Cifar10
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 and Caltech101 Training')
parser.add_argument('--dataset', default='Cifar10', type=str, help='dataset (Cifar10 or Imbalanced_Cifar10 or Caltech101)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cycles', default=10, type=int, help='number of active learning cycles')
parser.add_argument('--sample_ratio', default=0.2, type=float, help='sampling ratio for each cycle')
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

    trainset = Loader_Cifar10(is_train=True, transform=transform_train)
    train_length = len(trainset)
    initial_samples = train_length // 50

    indices = list(range(train_length))
    random.shuffle(indices)
    labeled_set = indices[:initial_samples]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

    testset = Loader_Cifar10(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
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

    trainset = Loader3_Cifar10(is_train=True, transform=transform_train)
    train_length = len(trainset)
    initial_samples = train_length // 50

    indices = list(range(train_length))
    random.shuffle(indices)
    labeled_set = indices[:initial_samples]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

    testset = Loader3_Cifar10(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


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

    trainset = Loader_Caltech101(is_train=True, transform=transform_train)
    train_length = len(trainset)
    initial_samples = train_length // 50

    indices = list(range(train_length))
    random.shuffle(indices)
    labeled_set = indices[:initial_samples]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

    testset = Loader_Caltech101(is_train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    net = ResNet_Caltech101()
    num_classes = testset.num_classes
    net.linear = nn.Linear(net.linear.in_features, num_classes)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'checkpoint_{args.dataset}'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint_{args.dataset}/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

# Training
def train(epoch):
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

#Test
def test(epoch):
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
        if not os.path.isdir(f'checkpoint_{args.dataset}'):
            os.mkdir(f'checkpoint_{args.dataset}')
        torch.save(state, f'./checkpoint_{args.dataset}/ckpt.pth')
        best_acc = acc

# Active Learning Cycles
print(f'>> Initial labeled length: {len(labeled_set)}')

for cycle in range(args.cycles):
    print(f'Cycle {cycle}/{args.cycles}')


    # Training with the current labeled set
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        test(epoch)
        scheduler.step()
    with open(f'./main_best_{args.dataset}_random.txt', 'a') as f:
        f.write(str(cycle) + ' ' + str(best_acc)+'\n')

    # Add new samples to the labeled set based on random sampling
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    
    # Number of new samples to add is the same as the initial labeled set size
    num_new_samples = initial_samples

    # Ensure that we do not resample already labeled data
    current_labeled_set = set(labeled_set)
    new_samples = [index for index in indices if index not in current_labeled_set][:num_new_samples]
    
    labeled_set.extend(new_samples)
    print(f'>> Labeled length: {len(labeled_set)}')

    # Update trainloader with new labeled set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, sampler=SubsetRandomSampler(labeled_set))