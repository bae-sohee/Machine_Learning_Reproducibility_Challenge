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
from torchvision.transforms import ToTensor
import shutil
import numpy as np

'''Cifar 10'''

class Cifar10_SupervisedLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Cifar10/DATA'):
        self.is_train = is_train
        self.transform = transform
        if self.is_train:  
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:  
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(img_path.split('/')[-2])
        
        if self.is_train:
            return img, label, img_path
        else:
            return img, label

class Imbalanced_Cifar10_SupervisedLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Imbalanced_Cifar10/DATA'):  
        self.is_train = is_train
        self.transform = transform
        if self.is_train:  
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:  
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(img_path.split('/')[-2])
        
        if self.is_train:
            return img, label, img_path
        else:
            return img, label

        
class Loader_Cifar10(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Cifar10/DATA'):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        if self.is_train: 
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        return img, label

class Loader2_Cifar10(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Cifar10/DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        if self.is_train:
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
            else:
                self.img_path = path_list

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        return img, label
    
    
class Loader3_Cifar10(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Imbalanced_Cifar10/DATA'):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        if self.is_train: 
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        return img, label
    
        
    
class Loader4_Cifar10(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Imbalanced_Cifar10/DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        if self.is_train:
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
            else:
                self.img_path = path_list

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        return img, label

    

'''Caltech101'''

class Caltech101_SupervisedLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Caltech101/DATA'):
        self.is_train = is_train
        self.transform = transform
        self.class_to_idx = self._find_classes(path)
        self.num_classes = len(self.class_to_idx)
        if self.is_train:  
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:  
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def _find_classes(self, path):
        classes = sorted(entry.name for entry in os.scandir(os.path.join(path, 'train')) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        class_name = img_path.split(os.sep)[-2]
        label = self.class_to_idx[class_name]
        
        if self.is_train:
            return img, label, img_path
        else:
            return img, label

class Loader_Caltech101(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Caltech101/DATA'):
        self.is_train = is_train
        self.transform = transform
        self.class_to_idx = self._find_classes(path)
        self.num_classes = len(self.class_to_idx)
        if self.is_train: 
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def _find_classes(self, path):
        classes = sorted(entry.name for entry in os.scandir(os.path.join(path, 'train')) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Error reading image at path: {img_path}")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            img = self.transform(img)
        class_name = img_path.split(os.sep)[-2]
        label = self.class_to_idx[class_name]
        return img, label


class Loader2_Caltech101(Dataset):
    def __init__(self, is_train=True, transform=None, path='./Caltech101/DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.class_to_idx = self._find_classes(path)
        if self.is_train:
            self.img_path = [os.path.join(path, p.strip().replace('Caltech101/DATA/', '').replace('./', '')) for p in path_list]
        else:
            if path_list is None:
                self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
            else:
                self.img_path = [os.path.join(path, p.strip().replace('Caltech101/DATA/', '').replace('./', '')) for p in path_list]

    def _find_classes(self, path):
        classes = sorted(entry.name for entry in os.scandir(os.path.join(path, 'train')) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            # Return a default image and label if the image is unreadable.
            img = Image.new('RGB', (224, 224))  # Create a black image of the required size
            img = ToTensor()(img)  # Convert to tensor
            label = 0  # Assign a default label, or handle as appropriate
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = ToTensor()(img)  # Ensure the image is converted to a tensor
            class_name = img_path.split(os.sep)[-2]
            label = self.class_to_idx[class_name]
        return img, label
