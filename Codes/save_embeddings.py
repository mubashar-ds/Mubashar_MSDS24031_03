import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from dataset import dataset_splits

from model import MyEmbeddingNetwork

def saving_embeddings(data_path, model_path, save_prefix, split = 'test'):

    os.makedirs('../Embeddings' , exist_ok = True)

    device = 'cpu'

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = datasets.ImageFolder(data_path, transform)
    
    training_dataset, validation_dataset, testing_dataset = dataset_splits(dataset)

    if split == 'train':
        selected_dataset = training_dataset
    elif split == 'validation':
        selected_dataset = validation_dataset
    else:
        selected_dataset = testing_dataset

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

            images_numpy = images.permute(0, 2, 3, 1).numpy()
            images_numpy = (images_numpy * 255).astype(np.uint8)
            the_images.append(images_numpy)

            the_labels.append(labels)

            the_embeddings.append(embedding)
    
    the_images = np.concatenate(the_images)
    the_labels = torch.cat(the_labels).numpy()
    the_embeddings = torch.cat(the_embeddings).numpy()

    np.save(f'embeddings/{save_prefix}_{split}_images.npy', the_images)
    np.save(f'embeddings/{save_prefix}_{split}_labels.npy', the_labels)
    np.save(f'embeddings/{save_prefix}_{split}_embeddings.npy', the_embeddings)

