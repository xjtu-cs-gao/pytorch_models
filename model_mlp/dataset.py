import torch.utils.data as data
import torch
import numpy as np

from __main__ import SEED
np.random.seed(SEED)

import opt


class MyDataset(data.Dataset):

    def __init__(self, image_dir, label_dir, transform=False):
        super(MyDataset, self).__init__()
        self.images = np.load(image_dir)
        self.labels = np.load(label_dir)
        
        self.transform = transform

    def __getitem__(self, index):
        input = self.images[index]
        target = self.labels[index]

        if self.transform:
            input = torch.FloatTensor(input / 255.)
            target = torch.FloatTensor(target)

        return input, target

    def __len__(self):
        return len(self.images)
