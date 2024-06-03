import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
from models import *
from rotation_loader import Cifar10_RotationLoader, Imbalanced_Cifar10_RotationLoader, Caltech101_RotationLoader
from utils import progress_bar

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Training with Rotation Task')
parser.add_argument('--dataset', required=True, type=str, help='Dataset to use: Cifar10, Imbalanced_Cifar10, Caltech101')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 1
train_batch_size = 256
test_batch_size = 100

# Prepare dataset and model
print("Prepare dataset")
if args.dataset == 'Cifar10':
    rotation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainloader = DataLoader(Cifar10_RotationLoader(is_train=True, transform=rotation_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Cifar10_RotationLoader(is_train=False, transform=rotation_transform), batch_size=test_batch_size, shuffle=False, num_workers=4)
    net = ResNet18()
    model_filename = 'rotation_Cifar10.pth'
    log_filename = 'best_rotation_Cifar10.txt'
elif args.dataset == 'Imbalanced_Cifar10':
    rotation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainloader = DataLoader(Imbalanced_Cifar10_RotationLoader(is_train=True, transform=rotation_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Imbalanced_Cifar10_RotationLoader(is_train=False, transform=rotation_transform), batch_size=test_batch_size, shuffle=False, num_workers=4)
    net = ResNet18()
    model_filename = 'rotation_Imbalance_Cifar10.pth'
    log_filename = 'best_rotation_Imbalanced_Cifar10.txt'
elif args.dataset == 'Caltech101':
    rotation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainloader = DataLoader(Caltech101_RotationLoader(is_train=True, transform=rotation_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Caltech101_RotationLoader(is_train=False, transform=rotation_transform), batch_size=test_batch_size, shuffle=False, num_workers=4)
    net = ResNet_Caltech101()
    model_filename = 'rotation_Caltech101.pth'
    log_filename = 'best_rotation_Caltech101.txt'
else:
    raise ValueError("Unsupported dataset. Choose from: Cifar10, Imbalanced_Cifar10, Caltech101")

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

# Training function
def train(epoch):
    print('==> Model Training..')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(trainloader):
        inputs, inputs1, inputs2, inputs3 = inputs.to(device), inputs1.to(device), inputs2.to(device), inputs3.to(device)
        targets, targets1, targets2, targets3 = targets.to(device), targets1.to(device), targets2.to(device), targets3.to(device)
        optimizer.zero_grad()
        outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs1, targets1)
        loss3 = criterion(outputs2, targets2)
        loss4 = criterion(outputs3, targets3)
        loss = (loss1 + loss2 + loss3 + loss4) / 4.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        _, predicted3 = outputs3.max(1)
        total += targets.size(0) * 4

        correct += predicted.eq(targets).sum().item()
        correct += predicted1.eq(targets1).sum().item()
        correct += predicted2.eq(targets2).sum().item()
        correct += predicted3.eq(targets3).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

# Testing function
def test(epoch):
    print('==> Model Testing..')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, inputs2, inputs3 = inputs.to(device), inputs1.to(device), inputs2.to(device), inputs3.to(device)
            targets, targets1, targets2, targets3 = targets.to(device), targets1.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1 + loss2 + loss3 + loss4) / 4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0) * 4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    with open(f'./{log_filename}', 'a') as f:
        f.write(str(acc) + ':' + str(epoch) + '\n')

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_dir = './checkpoint'
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)  # Create necessary parent directories
        torch.save(state, os.path.join(checkpoint_dir, model_filename))
        best_acc = acc

# Training and testing loop
for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
