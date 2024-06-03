import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

'''Cifar10'''

class Cifar10_ColorizationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, grayscale_transform=None):
        self.is_train = is_train
        self.transform = transform
        self.grayscale_transform = grayscale_transform
        if self.is_train:
            self.img_path = glob.glob('./Cifar10/DATA/train/*/*')
        else:
            self.img_path = glob.glob('./Cifar10/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        original_img = Image.open(self.img_path[idx]).convert('RGB') 
        grayscale_img = original_img.convert("L")
        grayscale_img = grayscale_img.convert("RGB") 
        
        if self.transform:
            original_img = self.transform(original_img)
        if self.grayscale_transform:
            grayscale_img = self.grayscale_transform(grayscale_img)
        return original_img, grayscale_img


# 컬러 이미지 변환
# color_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
# ])

# # 그레이스케일 이미지 변환
# grayscale_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Ensure 3 channels for grayscale
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),   # 그레이스케일 이미지를 위해 평균과 표준편차를 설정
# ])

# # 데이터 로더 초기화
# train_dataset = Cifar10_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# test_dataset = Cifar10_ColorizationLoader(is_train=False, transform=color_transform, grayscale_transform=grayscale_transform)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


'''Imabalanced_Cifar10'''

class Imbalanced_Cifar10_ColorizationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, grayscale_transform=None):
        self.is_train = is_train
        self.transform = transform
        self.grayscale_transform = grayscale_transform
        if self.is_train:
            self.img_path = glob.glob('./Imbalanced_Cifar10/DATA/train/*/*')
        else:
            self.img_path = glob.glob('./Imbalanced_Cifar10/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        original_img = Image.open(self.img_path[idx]).convert('RGB') 
        grayscale_img = original_img.convert("L")
        grayscale_img = grayscale_img.convert("RGB") 
        
        if self.transform:
            original_img = self.transform(original_img)
        if self.grayscale_transform:
            grayscale_img = self.grayscale_transform(grayscale_img)
        return original_img, grayscale_img
    
    
    

'''Caltech101'''


# Caltech101 Colorization Loader
class Caltech101_ColorizationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, grayscale_transform=None):
        self.is_train = is_train
        self.transform = transform
        self.grayscale_transform = grayscale_transform
        if self.is_train:
            self.img_path = glob.glob('./Caltech101/DATA/train/*/*')
        else:
            self.img_path = glob.glob('./Caltech101/DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        original_img = Image.open(self.img_path[idx]).convert('RGB') 
        grayscale_img = original_img.convert("L")
        grayscale_img = grayscale_img.convert("RGB") 
        
        if self.transform:
            original_img = self.transform(original_img)
        if self.grayscale_transform:
            grayscale_img = self.grayscale_transform(grayscale_img)
        return original_img, grayscale_img

# # 컬러 이미지 변환
# color_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
# ])

# # 그레이스케일 이미지 변환
# grayscale_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Ensure 3 channels for grayscale
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),   # 그레이스케일 이미지를 위해 평균과 표준편차를 설정
# ])

# # 데이터 로더 초기화
# train_dataset = Cifar10_ColorizationLoader(is_train=True, transform=color_transform, grayscale_transform=grayscale_transform)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# test_dataset = Cifar10_ColorizationLoader(is_train=False, transform=color_transform, grayscale_transform=grayscale_transform)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)