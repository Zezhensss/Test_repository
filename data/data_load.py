import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os import listdir
from os.path import isfile, join

class Imagenet(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [join(self.root_dir, f) for f in listdir(self.root_dir)]

    def __getitem__(self, index):
        image = Image.open(self.files[index])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.files)

