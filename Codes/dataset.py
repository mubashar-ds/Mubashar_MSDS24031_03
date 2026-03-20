import os

import torch
from torchvision import datasets
from torch.utils.data import Dataset

import random
from PIL import Image
class MyContrastiveDataset(Dataset):
    def __init__(self, root_directory, transform = None):
        self.dataset = datasets.ImageFolder(root = root_directory, transform= transform)
        self.transform = transform

        # maping class to indexes...
        self.class_to_indexes = {}
        for indx, (_, label) in enumerate(self.dataset.samples):
            if label not in self.class_to_indexes:
                self.class_to_indexes[label] = []
            self.class_to_indexes[label].append(indx)

    def __getitem__(self, index):
        image_1, label_1 = self.dataset[index]

        # negative pair ..
        if random.random() > 0.5:
            label_2 = label_1
            while label_2 == label_1:
                index_2 = random.randint(0, len(self.dataset) - 1)
                _, label_2 = self.dataset[index_2]
            image_2, _ = self.dataset[index_2]
            label = 0
            
        # positive pair ...
        else:
            # positive pair ...
            index_2 = index
            while index_2 == index:
                index_2 = random.choice(self.class_to_indexes[label_1])
            image_2, _ = self.dataset[index_2]
            label = 1

        return image_1, image_2, torch.tensor(label, dtype = torch.float32)

    def __len__(self):
        return len(self.dataset)

# if __name__ == '__main__':
#     from torchvision import transforms

#     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     data_path = './Dataset/caltech-101'

#     print('testinig my contrastive dataset ...')
#     contrastive = MyContrastiveDataset(data_path, transform)
#     image_1, image_2, label = contrastive[5]
#     print('label : ', label)
#     print('image_1 shape : ', image_1.shape)
#     print('image_2 shape : ', image_2.shape)