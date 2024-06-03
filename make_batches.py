import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np

from models import ResNet18_colorization, ResNet_colorization_Caltech101, ResNet18, ResNet_Caltech101
from colorization_loader import Cifar10_ColorizationLoader, Imbalanced_Cifar10_ColorizationLoader, Caltech101_ColorizationLoader
from rotation_loader import Cifar10_RotationLoader, Imbalanced_Cifar10_RotationLoader, Caltech101_RotationLoader
from utils import progress_bar


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Maker')
    parser.add_argument('--dataset', choices=['Cifar10', 'Imbalanced_Cifar10', 'Caltech101'], required=True, help='Dataset to use')
    parser.add_argument('--task', choices=['rotation', 'colorization'], required=True, help='Task to perform')
    args = parser.parse_args()
    return args


def prepare_data(dataset, task):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if task == 'colorization':
        grayscale_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        grayscale_transform = None

    if dataset == 'Cifar10':
        if task == 'colorization':
            data_loader = Cifar10_ColorizationLoader(is_train=True, transform=transform, grayscale_transform=grayscale_transform)
        else:
            data_loader = Cifar10_RotationLoader(is_train=True, transform=transform)
    elif dataset == 'Imbalanced_Cifar10':
        if task == 'colorization':
            data_loader = Imbalanced_Cifar10_ColorizationLoader(is_train=True, transform=transform, grayscale_transform=grayscale_transform)
        else:
            data_loader = Imbalanced_Cifar10_RotationLoader(is_train=True, transform=transform)
    else:
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
        if task == 'colorization':
            data_loader = Caltech101_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform)
        else:
            data_loader = Caltech101_RotationLoader(is_train=True, transform=color_transform)

    return DataLoader(data_loader, batch_size=1, shuffle=False, num_workers=4)


# def load_model(dataset, task, device):
#     if task == 'colorization':
#         if dataset == 'Caltech101':
#             model = ResNet_colorization_Caltech101().to(device)
#         else:
#             model = ResNet18_colorization().to(device)
#         checkpoint_path = f'./checkpoint/colorization_{dataset}.pth'
#     else:
#         if dataset == 'Caltech101':
#             model = ResNet_Caltech101().to(device)
#         else:
#             model = ResNet18().to(device)
#         checkpoint_path = f'./checkpoint/rotation_{dataset}.pth'

#     try:
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['net'])
#     except RuntimeError as e:
#         print(f"Error loading state_dict for model from checkpoint {checkpoint_path}: {e}")
#         raise e

#     if device == 'cuda':
#         model = torch.nn.DataParallel(model)
#         cudnn.benchmark = True
#     return model

def load_model(dataset, task, device):
    if task == 'colorization':
        if dataset == 'Caltech101':
            model = ResNet_colorization_Caltech101().to(device)
        else:
            model = ResNet18_colorization().to(device)
        checkpoint_path = f'./checkpoint/colorization_{dataset}.pth'
    else:
        if dataset == 'Caltech101':
            model = ResNet_Caltech101().to(device)
        else:
            model = ResNet18().to(device)
        checkpoint_path = f'./checkpoint/rotation_{dataset}.pth'

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])

    return model


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testloader = prepare_data(args.dataset, args.task)
    model = load_model(args.dataset, args.task, device)
    criterion = nn.MSELoss() if args.task == 'colorization' else nn.CrossEntropyLoss()
    result_file = f'./{args.task}_loss_{args.dataset}.txt'
    batch_folder = f'./loss_{args.dataset}_{args.task}'

    def test():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            with open(result_file, 'a') as f:
                for batch_idx, data in enumerate(testloader):
                    if args.task == 'colorization':
                        color, grayscale = data
                        color, grayscale = color.to(device), grayscale.to(device)
                        output = model(grayscale)
                        loss = criterion(output, color)
                    else:
                        inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path = data
                        inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
                        inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
                        outputs = model(inputs)
                        outputs1 = model(inputs1)
                        outputs2 = model(inputs2)
                        outputs3 = model(inputs3)
                        loss1 = criterion(outputs, targets)
                        loss2 = criterion(outputs1, targets1)
                        loss3 = criterion(outputs2, targets2)
                        loss4 = criterion(outputs3, targets3)
                        loss = (loss1 + loss2 + loss3 + loss4) / 4.
                    test_loss += loss.item()
                    loss_value = loss.item()
                    path = testloader.dataset.img_path[batch_idx * testloader.batch_size: (batch_idx + 1) * testloader.batch_size]
                    for p in path:
                        s = str(float(loss_value)) + '_' + p + "\n"
                        f.write(s)
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f' % (test_loss / (batch_idx + 1)))

    test()

    with open(result_file, 'r') as f:
        losses = f.readlines()

    loss_values = []
    file_paths = []

    for line in losses:
        line = line.strip()
        split_index = line.find('_')
        loss_value = line[:split_index]
        file_path = line[split_index+1:]
        loss_values.append(loss_value)
        file_paths.append(file_path)

    loss_array = np.array(loss_values, dtype=float)
    sort_index = np.argsort(loss_array)[::-1]

    if not os.path.isdir(batch_folder):
        os.mkdir(batch_folder)

    batch_numbers = 10
    sample_numbers = int(len(testloader) / batch_numbers)
    for i in range(batch_numbers):
        sample = sort_index[i * sample_numbers:(i + 1) * sample_numbers]
        batch_file = f'{batch_folder}/batch_{i}.txt'
        with open(batch_file, 'w') as f:
            for idx in sample:
                f.write(file_paths[idx] + '\n')


if __name__ == "__main__":
    main()
