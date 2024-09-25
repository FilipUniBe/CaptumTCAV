import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL
import torch.nn as nn
from PIL import Image


class Load_from_path_Dataset(Dataset):
    def __init__(self, img_paths=None, home_dir=None, y=None, dim1=320, dim2=320, aug=True, mode="test",
                 return_id=False):
        self.img_labels = y
        self.img_dir = home_dir
        self.img_paths = img_paths
        self.dim1 = dim1
        self.dim2 = dim2
        self.mode = mode
        self.resizing = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.dim1, self.dim2), antialias=True)
                                            ])

    def __len__(self):
        return len(self.img_labels)

    def normalising(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    def transformation(self, image):
        image = np.float32(np.array(image))

        image = self.normalising(image)
        image = self.resizing(image)

        image = image / 255.

        return image

    def __getitem__(self, idx):
        # print("self.img_dir",self.img_dir)
        # print("self.img_paths[idx]",self.img_paths[idx])
        img_path = os.path.join(self.img_dir + self.img_paths[idx])
        image = Image.open(img_path).convert("L")
        image = self.transformation(image)
        if image.shape != (1, self.dim1, self.dim2):
            image = image.unsqueeze(0)
        label = self.img_labels[idx]
        label = torch.from_numpy(label)
        return image, label