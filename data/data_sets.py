"""
Dataset in charge of knowing where the source of data, labels and transforms.
Should provide access to the data by indexing.
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch.utils.data as data
import cv2


class FolderDataset(data.Dataset):

    def __init__(self, data):        
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        elem = self.data.iloc[index].values
        user, item = elem[:2]

        gen = np.array(elem[2].split(',')).astype(int)

        target = elem[-1]
        return user, item, gen, target

if __name__ == '__main__':

    pass










