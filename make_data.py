import glob
import random
import cv2
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms



'----------------------------------------------------------------------------------------------------------------------'
'''Cifar10'''


class Cifar10SaveDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.split = split

    def __getitem__(self, i):
        x, y = self.dataset[i]
        path = f'./Cifar10/DATA/{self.split}/{y}/{i}.png'  # 파일 저장 이름

        if not os.path.isdir(f'./Cifar10/DATA/{self.split}/{y}'):  # 폴더가 없으면 폴더 생성
            os.makedirs(f'./Cifar10/DATA/{self.split}/{y}')

        if isinstance(x, np.ndarray):
            img = Image.fromarray(x)
        else:
            img = x

        img.save(path)  # path에 저장
        return x, y

    def __len__(self):
        return len(self.dataset)

def Cifar10_save_dataset():
    # 데이터 다운로드
    print('Cifar10 loading..')
    trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Cifar10/data', train=True, download=True, transform=None)
    testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Cifar10/data', train=False, download=True, transform=None)

    train_dataset = Cifar10SaveDataset(trainset_CIFAR10, split='train')  # 50,000개
    test_dataset = Cifar10SaveDataset(testset_CIFAR10, split='test')  # 10,000개

    # 폴더 생성
    os.makedirs('./Cifar10/DATA/train', exist_ok=True)
    os.makedirs('./Cifar10/DATA/test', exist_ok=True)

    # 이미지 저장
    for idx in range(len(train_dataset)):
        train_dataset[idx]

    for idx in range(len(test_dataset)):
        test_dataset[idx]

    print('Cifar10 saved')



'----------------------------------------------------------------------------------------------------------------------'
'''imbalanced Cifar10'''



class ImbalancedCifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split='train', class_imbalance=None):
        self.dataset = dataset
        self.split = split
        self.class_imbalance = class_imbalance
        self.class_counts = {k: 0 for k in class_imbalance.keys()}
        self.imbalanced = self.create_imbalanced()

    # 불균형 함수
    def create_imbalanced(self):
        indices = []
        class_counts = {k: 0 for k in self.class_imbalance.keys()}
        for idx, (x, y) in enumerate(self.dataset):
            if class_counts[y] < self.class_imbalance[y]:
                indices.append(idx)
                class_counts[y] += 1
        return indices

    def __getitem__(self, i):
        idx = self.imbalanced[i]
        x, y = self.dataset[idx]
        path = f'./Imbalanced_Cifar10/DATA/{self.split}/{y}/{i}.png'

        # 필요한 디렉토리 생성
        if not os.path.isdir(f'./Imbalanced_Cifar10/DATA/{self.split}/{y}'):
            os.makedirs(f'./Imbalanced_Cifar10/DATA/{self.split}/{y}')

        # 이미지 저장
        if isinstance(x, np.ndarray):
            img = Image.fromarray(x)
        else:
            img = x

        img.save(path)
        return x, y

    def __len__(self):
        return len(self.imbalanced)

def Imbalanced_Cifar10_save_dataset():
    # 데이터 다운로드
    print('Imbalanced Cifar10 loading...')
    trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Imbalanced_Cifar10/data', train=True, download=True, transform=None)
    testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Imbalanced_Cifar10/data', train=False, download=True, transform=None)

    # 클래스 불균형 설정
    class_imbalance_train = {
        0: 500,    # 비행기
        1: 1000,   # 자동차
        2: 1500,   # 새
        3: 2000,   # 고양이
        4: 2500,   # 사슴
        5: 3000,   # 개
        6: 3500,   # 개구리
        7: 4000,   # 말
        8: 4500,   # 배
        9: 5000    # 트럭
    }

    class_imbalance_test = {
        0: 1000,   # 비행기
        1: 1000,   # 자동차
        2: 1000,   # 새
        3: 1000,   # 고양이
        4: 1000,   # 사슴
        5: 1000,   # 개
        6: 1000,   # 개구리
        7: 1000,   # 말
        8: 1000,   # 배
        9: 1000    # 트럭
    }

    train_dataset = ImbalancedCifar10Dataset(trainset_CIFAR10, split='train', class_imbalance=class_imbalance_train)
    test_dataset = ImbalancedCifar10Dataset(testset_CIFAR10, split='test', class_imbalance=class_imbalance_test)

    # 폴더 생성
    os.makedirs('./Imbalanced_Cifar10/DATA/train', exist_ok=True)
    os.makedirs('./Imbalanced_Cifar10/DATA/test', exist_ok=True)

    # 이미지 저장
    for idx in range(len(train_dataset)):
        train_dataset[idx]

    for idx in range(len(test_dataset)):
        test_dataset[idx]

    print('Imbalanced Cifar10 saved')



'----------------------------------------------------------------------------------------------------------------------'
'''Caltech101'''



def Caltech101_save_dataset(root_dir="./101_ObjectCategories", train_dir="./Caltech101/DATA/train", test_dir="./Caltech101/DATA/test", split_ratio=0.8):
    # 저장할 폴더 생성
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 제외할 폴더 목록
    exclude_folders = {'BACKGROUND_Google', 'caltech101'}

    for label in os.listdir(root_dir):
        if label in exclude_folders:
            continue
        
        label_dir = os.path.join(root_dir, label)  # 폴더명 순환 label_dir = 디렉토리/label폴더명
        if os.path.isdir(label_dir):
            images = os.listdir(label_dir)  # 이미지 파일 저장
            random.shuffle(images)  # 랜덤으로 추출
            split_idx = int(len(images) * split_ratio)  # train/test index 지정

            train_label_dir = os.path.join(train_dir, label)
            test_label_dir = os.path.join(test_dir, label)

            os.makedirs(train_label_dir, exist_ok=True)  # exist_ok : 디렉토리 존재하면 넘어가기
            os.makedirs(test_label_dir, exist_ok=True)

            for img in images[:split_idx]:
                src_path = os.path.join(label_dir, img)
                dst_path = os.path.join(train_label_dir, img)
                shutil.copy(src_path, dst_path)

            for img in images[split_idx:]:
                src_path = os.path.join(label_dir, img)
                dst_path = os.path.join(test_label_dir, img)
                shutil.copy(src_path, dst_path)

    print('Caltech101 saved')


# 함수 호출

Cifar10_save_dataset()
Imbalanced_Cifar10_save_dataset()
Caltech101_save_dataset()

