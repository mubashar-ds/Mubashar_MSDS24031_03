import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from dataset import dataset_splits

from model import MyEmbeddingNetwork

import argparse

from torch.utils.data import Subset
import random

def saving_embeddings(data_path, model_path, save_prefix, split):

    os.makedirs('../Embeddings' , exist_ok = True)

    device = 'cpu'

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    base_dataset = datasets.ImageFolder(args.data_path, transform)

    subset_size = 3000
    indices = random.sample(range(len(base_dataset)), subset_size)
    base_dataset = Subset(base_dataset, indices)

    training_ds, validation_ds, testing_ds = dataset_splits(base_dataset)

    if split == 'train':
        selected_dataset = training_ds

    elif split == 'validation':
        selected_dataset = validation_ds

    else:
        selected_dataset = testing_ds

    data_loader = DataLoader(selected_dataset, batch_size = 8)
 
    model = MyEmbeddingNetwork()
    
    model.load_state_dict(torch.load(model_path))

    model.eval()

    the_images = []
    the_labels = []
    the_embeddings = []

    with torch.no_grad():

        for images, labels in data_loader:

            embedding = model(images)

            # for making it memory efficient..
            
            images_numpy = images.permute(0, 2, 3, 1).numpy()
            images_numpy = (images_numpy * 255).astype(np.uint8)
            the_images.append(images_numpy)

            the_labels.append(labels)

            the_embeddings.append(embedding)
    
    the_images = np.concatenate(the_images)
    the_labels = torch.cat(the_labels).numpy()
    the_embeddings = torch.cat(the_embeddings).numpy()

    np.save(f'../Embeddings/{save_prefix}_{split}_images.npy', the_images)
    np.save(f'../Embeddings/{save_prefix}_{split}_labels.npy', the_labels)
    np.save(f'../Embeddings/{save_prefix}_{split}_embeddings.npy', the_embeddings)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--model_path', type = str, required= True)
    parser.add_argument('--mode', type = str, required =True, choices = ['contrastive', 'triplet', 'hard'])

    args = parser.parse_args()

    for split in ['train', 'validation', 'test']:
        saving_embeddings(args.data_path, args.model_path, args.mode, split)