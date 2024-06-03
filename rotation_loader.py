import glob
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import shutil


# # 데이터 download
# # torchvision 모듈 사용 - CIFAR10
# trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Cifar10/data', train=True, download=True, transform=None)
# testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Cifar10/data', train=False, download=True, transform=None)

# train_dataset = save_dataset(trainset, split='train')
# test_dataset = save_dataset(testset, split='test')
'''Cifar10'''

# 기존 코드 출력 부분은 주석 걸어놓을게여
# 폴더 생성

# if not os.path.isdir('./Cifar10/DATA'):
#     os.mkdir('./Cifar10/DATA')

# if not os.path.isdir('./Cifar10/DATA/train'):
#     os.mkdir('./Cifar10/DATA/train')

# if not os.path.isdir('./Cifar10/DATA/test'):
#     os.mkdir('./Cifar10/DATA/test')

# for idx, i in enumerate(train_dataset):
#     train_dataset[idx]
#     # print(idx)

# for idx, i in enumerate(test_dataset):
#     test_dataset[idx]
#     # print(idx)

    
    
# Cifar10 Rotation Loader    
class Cifar10_RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train: # train
            self.img_path = glob.glob('./Cifar10/DATA/train/*/*') # 하위 폴더 전체 데려오기
        else:
            self.img_path = glob.glob('./Cifar10/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img_path = self.img_path[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        # 이미지 형식 맞춰주기. BGR형식이 일반적이라 안 빼고 넣어놓음
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        img1 = torch.rot90(img, 1, [1, 2])
        img2 = torch.rot90(img, 2, [1, 2])
        img3 = torch.rot90(img, 3, [1, 2])
        imgs = [img, img1, img2, img3]
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)

        if self.is_train:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]
        else:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]


# Cifar10_rotation_transform = transforms.Compose([
#     # transforms.Resize((128, 128)),  # Resize to 128x128 for better visualization
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

Cifar10_rotation_transform = transforms.Compose([
    # transforms.Resize((128, 128)),  # Resize to 128x128 for better visualization
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

Cifar10_Rotation = Cifar10_RotationLoader(is_train=True, transform=Cifar10_rotation_transform)
trainloader = DataLoader(Cifar10_Rotation, batch_size=1, shuffle=True, num_workers=4)


'----------------------------------------------------------------------------------------------------------------------'
'''imbalanced Cifar10'''

import os
import torch
import torchvision
from PIL import Image
import numpy as np

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

# 데이터 다운로드
trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Imbalanced_Cifar10/data', train=True, download=True, transform=None)
testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./Imbalanced_Cifar10/data', train=False, download=True, transform=None)

# 클래스 불균형 설정
class_imbalance = {
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

# train_dataset = ImbalancedCifar10Dataset(trainset_CIFAR10, split='train', class_imbalance=class_imbalance)
# test_dataset = ImbalancedCifar10Dataset(testset_CIFAR10, split='test', class_imbalance=class_imbalance)

# 폴더 생성
# if not os.path.isdir('./Imbalanced_Cifar10/DATA'):
#     os.mkdir('./Imbalanced_Cifar10/DATA')

# if not os.path.isdir('./Imbalanced_Cifar10/DATA/train'):
#     os.mkdir('./Imbalanced_Cifar10/DATA/train')

# if not os.path.isdir('./Imbalanced_Cifar10/DATA/test'):
#     os.mkdir('./Imbalanced_Cifar10/DATA/test')

# # 이미지 저장
# for idx in range(len(train_dataset)):
#     train_dataset[idx]

# for idx in range(len(test_dataset)):
#     test_dataset[idx]

# print('Imbalanced Cifar10 saved')


# Imbalanced Cifar10 Rotation Loader    
class Imbalanced_Cifar10_RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train: # train
            self.img_path = glob.glob('./Imbalanced_Cifar10/DATA/train/*/*') # 하위 폴더 전체 데려오기
        else:
            self.img_path = glob.glob('./Imbalanced_Cifar10/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img_path = self.img_path[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        # 이미지 형식 맞춰주기. BGR형식이 일반적이라 안 빼고 넣어놓음
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        img1 = torch.rot90(img, 1, [1, 2])
        img2 = torch.rot90(img, 2, [1, 2])
        img3 = torch.rot90(img, 3, [1, 2])
        imgs = [img, img1, img2, img3]
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)

        if self.is_train:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]
        else:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]


# Cifar10_rotation_transform = transforms.Compose([
#     # transforms.Resize((128, 128)),  # Resize to 128x128 for better visualization
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# Cifar10_rotation_transform = transforms.Compose([
#     # transforms.Resize((128, 128)),  # Resize to 128x128 for better visualization
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# Cifar10_Rotation = Cifar10_RotationLoader(is_train=True, transform=Cifar10_rotation_transform)
# trainloader = DataLoader(Cifar10_Rotation, batch_size=1, shuffle=True, num_workers=4)

'''Caltech101'''


# # dataset make
# def Caltech101_save_dataset(root_dir, train_dir, test_dir, split_ratio=0.8):
#     # 저장할 폴더 생성
#     if not os.path.exists(train_dir):
#         os.makedirs(train_dir)
#     if not os.path.exists(test_dir):
#         os.makedirs(test_dir)

#     for label in os.listdir(root_dir):
#         label_dir = os.path.join(root_dir, label) # 폴더명 순환 label_dir = 디렉토리/label폴더명
#         if os.path.isdir(label_dir):
#             images = os.listdir(label_dir) # 이미지 파일 저장
#             random.shuffle(images) # 랜덤으로 추출
#             split_idx = int(len(images) * split_ratio) # train/test index 지정

#             train_label_dir = os.path.join(train_dir, label)
#             test_label_dir = os.path.join(test_dir, label)

#             os.makedirs(train_label_dir, exist_ok=True) # exist_ok : 디렉토리 존재하면 넘어가기
#             os.makedirs(test_label_dir, exist_ok=True)

#             for img in images[:split_idx]:
#                 src_path = os.path.join(label_dir, img)
#                 dst_path = os.path.join(train_label_dir, img)
#                 shutil.copy(src_path, dst_path)

#             for img in images[split_idx:]:
#                 src_path = os.path.join(label_dir, img)
#                 dst_path = os.path.join(test_label_dir, img)
#                 shutil.copy(src_path, dst_path)

# # 데이터셋 폴더 경로
# root_dir = "./101_ObjectCategories"

# # train/test 저장 경로
# train_dir = "./Caltect101/DATA/train"
# test_dir = "./Caltect101/DATA/test"

# 데이터 저장(폴더 생성, 데이터 저장 완료)
# Caltech101_save_dataset(root_dir, train_dir, test_dir)

# Caltech101 Rotation Loader
class Caltech101_RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            self.img_path = glob.glob('./Caltech101/DATA/train/*/*')
        else:
            self.img_path = glob.glob('./Caltech101/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img_path = self.img_path[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        # 이미지 형식 맞춰주기. BGR형식이 일반적이라 안 빼고 넣어놓음
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        img1 = torch.rot90(img, 1, [1, 2])
        img2 = torch.rot90(img, 2, [1, 2])
        img3 = torch.rot90(img, 3, [1, 2])
        imgs = [img, img1, img2, img3]
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)

        if self.is_train:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]
        else:
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]


    

# Caltech101_rotation_transform = transforms.Compose([
#     transforms.Resize((224, 224)), 
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# Caltech101_rotation_transform = transforms.Compose([
#     transforms.Resize((224, 224)), 
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# Caltech101_Rotation = Caltech101_RotationLoader(is_train=False, transform=Caltech101_rotation_transform)
# Caltech101_RotationLoader = DataLoader(Caltech101_Rotation, batch_size=1, shuffle=True)
# print(len(Caltech101_RotationLoader))
# for i in Caltech101_RotationLoader:
#     print(i[0].shape)
#     break


# # Caltech101 Colorization Loader
# class Caltech101_ColorizationLoader(Dataset):
#     def __init__(self, is_train=True, transform=None):
#         self.transform = transform
#         self.is_train = is_train
#         if self.is_train:
#             self.image_files = glob.glob('./Caltect101/DATA/train/*/*')
#         else:
#             self.image_files = glob.glob('./Caltect101/DATA/test/*/*')

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # original image(color)
#         original_img = Image.open(self.image_files[idx])
#         # grat image로 변환
#         grayscale_img = original_img.convert("L")
#         # 변환 적용
#         if self.transform:
#             original_img = self.transform(original_img)
#             grayscale_img = self.transform(grayscale_img)
#         return original_img, grayscale_img

# # 
# Caltech101_colorization_transform = transforms.Compose([
#     transforms.Resize((224, 224)), 
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
# rotation처럼 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 추가하면 시각화 할 때 오류 나서 우선 빼놓음

# Caltech101_colorization_dataset = Caltech101_ColorizationLoader(transform=Caltech101_colorization_transform)
# Caltech101_colorization_loader = DataLoader(Caltech101_colorization_dataset, batch_size=1, shuffle=True)
