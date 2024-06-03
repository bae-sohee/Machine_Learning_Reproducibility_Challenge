import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
from models import *
from colorization_loader import Cifar10_ColorizationLoader, Imbalanced_Cifar10_ColorizationLoader, Caltech101_ColorizationLoader
from utils import progress_bar

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Colorization Training')
parser.add_argument('--dataset', required=True, type=str, help='Dataset to use: Cifar10, Imbalanced_Cifar10, Caltech101')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 30
train_batch_size = 128
test_batch_size = 100

# Prepare dataset and model
if args.dataset == 'Cifar10':
    color_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    grayscale_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    trainloader = DataLoader(Cifar10_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Cifar10_ColorizationLoader(is_train=False, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=False, num_workers=4)
    net = ResNet18_colorization()
    model_filename = 'colorization_Cifar10.pth'
    log_filename = 'best_colorization_Cifar10.txt'
elif args.dataset == 'Imbalanced_Cifar10':
    color_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    grayscale_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    trainloader = DataLoader(Imbalanced_Cifar10_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Imbalanced_Cifar10_ColorizationLoader(is_train=False, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=False, num_workers=4)
    net = ResNet18_colorization()
    model_filename = 'colorization_Imbalanced_Cifar10.pth'
    log_filename = 'best_colorization_Imbalanced_Cifar10.txt'
elif args.dataset == 'Caltech101':
    color_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    grayscale_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    trainloader = DataLoader(Caltech101_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Caltech101_ColorizationLoader(is_train=False, transform=color_transform, grayscale_transform=grayscale_transform), batch_size=train_batch_size, shuffle=False, num_workers=4)
    net = ResNet_colorization_Caltech101()
    model_filename = 'colorization_Caltech101.pth'
    log_filename = 'best_colorization_Caltech101.txt'
else:
    raise ValueError("Unsupported dataset. Choose from: Cifar10, Imbalanced_Cifar10, Caltech101")

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

# Training function
def train(epoch):
    print('==> Model Training..')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (color, grayscale) in enumerate(trainloader):
        color, grayscale = color.to(device), grayscale.to(device)
        
        # Forward pass
        output = net(grayscale)
        
        optimizer.zero_grad()
        loss = criterion(output, color)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss / (batch_idx + 1)))

best_loss = float('inf')  # best test loss

# Testing function
def test(epoch):
    global best_loss  # Add this line
    print('==> Model Testing..')
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (color, grayscale) in enumerate(testloader):
            color, grayscale = color.to(device), grayscale.to(device)
            output = net(grayscale)
            loss = criterion(output, color)
            test_loss += loss.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f' % (test_loss / (batch_idx + 1)))

    # Save current loss and epoch to the log file
    with open(f'./{log_filename}', 'a') as f:
        f.write(str(test_loss) + ':' + str(epoch) + '\n')
        
    # Save checkpoint.
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        checkpoint_dir = './checkpoint'
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)  # Create necessary parent directories
        torch.save(state, os.path.join(checkpoint_dir, model_filename))
        best_loss = test_loss

# Training and testing loop
for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
