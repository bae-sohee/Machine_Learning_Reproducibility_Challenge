
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse

from models import ResNet18, ResNet_Caltech101, ResNet_Supervised_Caltech101
from supervised_loader import Cifar10_SupervisedLoader, Caltech101_SupervisedLoader, Imbalanced_Cifar10_SupervisedLoader

def main():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 and Caltech101 Training')
    parser.add_argument('--dataset', default='Cifar10', type=str, help='dataset (Cifar10 or Imbalanced_Cifar10 or Caltech101)')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    args = parser.parse_args()

    global best_acc 
    global best_epoch  

    dataset = args.dataset
    num_epochs = args.epochs
    batch_size = args.batch_size
    base_path = f'./{dataset}'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data preparation
    print('==> Preparing data..')
    
    if dataset == 'Cifar10' :
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = Cifar10_SupervisedLoader(is_train=True, transform=transform_train, path=os.path.join(base_path, 'DATA'))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = Cifar10_SupervisedLoader(is_train=False, transform=transform_test, path=os.path.join(base_path, 'DATA'))
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    elif dataset == 'Imbalanced_Cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = Imbalanced_Cifar10_SupervisedLoader(is_train=True, transform=transform_train, path=os.path.join(base_path, 'DATA'))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = Imbalanced_Cifar10_SupervisedLoader(is_train=False, transform=transform_test, path=os.path.join(base_path, 'DATA'))
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    else : 
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
        
        trainset = Caltech101_SupervisedLoader(is_train=True, transform=transform_train, path=os.path.join(base_path, 'DATA'))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = Caltech101_SupervisedLoader(is_train=False, transform=transform_test, path=os.path.join(base_path, 'DATA'))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    if dataset == 'Cifar10' or 'Imbalanced_Cifar10' :
        net = ResNet18()
    elif dataset == 'Caltech101':
        net = ResNet_Supervised_Caltech101(num_classes=trainset.num_classes)
    #     net = ResNet_Caltech101()
    #     num_classes = trainset.num_classes
    #     net.linear = nn.Linear(net.linear.in_features, num_classes)

    
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

    # Training
    def train(net, criterion, optimizer, epoch, trainloader):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets, paths) in enumerate(trainloader):
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

        avg_loss = train_loss / len(trainloader)
        accuracy = 100. * correct / total
        print(f'Train Loss: {train_loss/len(trainloader):.3f} | Train Acc: {accuracy:.3f}% ({correct}/{total})')
        return avg_loss

    # Testing
    def test(net, criterion, epoch):
        global best_acc
        global best_epoch

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

        avg_loss = test_loss / len(testloader)
        accuracy = 100. * correct / total
        print(f'Test Loss: {test_loss/len(testloader):.3f} | Test Acc: {accuracy:.3f}% ({correct}/{total})')

        if accuracy > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': accuracy,
                'epoch': epoch,
            }
            if not os.path.isdir(f'checkpoint_ideal_{dataset}'):
                os.mkdir(f'checkpoint_ideal_{dataset}')
            torch.save(state, f'./checkpoint_ideal_{dataset}/ckpt.pth')
            best_acc = accuracy
            best_epoch = epoch
        
        return avg_loss, accuracy

    # Main training loop
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss = train(net, criterion, optimizer, epoch, trainloader)
        test_loss, test_acc = test(net, criterion, epoch)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Load the best model and calculate loss for each train dataset
    checkpoint = torch.load(f'./checkpoint_ideal_{dataset}/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    if dataset == 'Cifar10' or 'Imbalanced_Cifar10':
        trainset = Cifar10_SupervisedLoader(is_train=True, transform=transform_test, path=os.path.join(base_path, 'DATA'))
    elif dataset == 'Caltech101':
        trainset = Caltech101_SupervisedLoader(is_train=True, transform=transform_test, path=os.path.join(base_path, 'DATA'))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

    net.eval()
    loss_records = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss_records.append(f"{loss.item()}_{paths[0]}")

    # Save loss values to a file
    with open(f'classification_loss_{dataset}.txt', 'w') as f:
        for record in loss_records:
            f.write(record + '\n')

    with open(f'./main_supervised_best_{dataset}.txt', 'a') as f:
        f.write(f'Epoch: {best_epoch} best accuracy: {best_acc}\n')

    # Plot train/test loss graph
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='tab:blue')
    plt.plot(test_losses, label='Test Loss', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.show()

    plt.savefig(f'Supervised learning loss {dataset}.png')

if __name__ == '__main__':
    main()
