import os

import random
import numpy as np

from torchvision import datasets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, required = True)
args = parser.parse_args()

random.seed(42)

os.makedirs('../Dataset', exist_ok = True)

dataset = datasets.ImageFolder(args.data_path)

subset_size = 5000

data_indices = random.sample(range(len(dataset)), subset_size)

np.save('../Dataset/subset-caltech-101.npy', data_indices)

print('Subset indices saved successfully...')