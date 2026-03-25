import os

import torch
from torchvision import datasets
from torch.utils.data import Dataset

import random
from PIL import Image
class MyContrastiveDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        # maping class to indexes ..

        self.class_to_indexes = {}
        for indx, (_, label) in enumerate(self.dataset):
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
            index_2 = index

            if len(self.class_to_indexes[label_1]) > 1:
                while index_2 == index:
                    index_2 = random.choice(self.class_to_indexes[label_1])

            image_2, _ = self.dataset[index_2]
            label = 1

        return image_1, image_2, torch.tensor(label, dtype = torch.float32)

    def __len__(self):
        return len(self.dataset)
    
class MyTripletDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        # mapping classes to indexes ...

        self.class_to_indexes = {}

        for indx, (_, label) in enumerate(self.dataset):
            if label not in self.class_to_indexes:
                self.class_to_indexes[label] = []
            self.class_to_indexes[label].append(indx)

        self.classes = list(self.class_to_indexes.keys())

    def __getitem__(self, index):

        anchor, label_anchor = self.dataset[index]

        # negative... differnet classs..

        negative_label = label_anchor

        while negative_label == label_anchor:
            negative_label = random.choice(self.classes)

        negative_index = random.choice(self.class_to_indexes[negative_label])
        negative, _ = self.dataset[negative_index]

        # positive.. same class and different image...

        positive_index = index

        if len(self.class_to_indexes[label_anchor]) > 1:
            while positive_index == index:
                positive_index = random.choice(self.class_to_indexes[label_anchor])

        positive, _ = self.dataset[positive_index]
        return anchor, positive, negative

import torch
from torch.utils.data import random_split

def dataset_splits(dataset):
    
    generator = torch.Generator().manual_seed(42)

    training_size = int(0.7 * len(dataset))
    validation_size = int(0.15 * len(dataset))

    testing_size = len(dataset) - training_size - validation_size

    return random_split(dataset,[training_size, validation_size, testing_size], generator = generator)

# if __name__ == '__main__':

#     data_path = '../Dataset/caltech-101'

#     from torchvision import transforms

#     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

#     print('testinig my contrastive dataset ...')

#     contrastive = MyContrastiveDataset(data_path, transform)
#     image_1, image_2, label = contrastive[5]
#     print('label : ', label)
#     print('image_1 shape : ', image_1.shape)
#     print('image_2 shape : ', image_2.shape)

#     print('\ntesting my triplet dataset...')

#     triplet = MyTripletDataset(data_path, transform)
#     anchor, negative, positive = triplet[2]
#     print('anchor shape : ', anchor.shape)
#     print('negative shape : ', negative.shape)
#     print('positive shape : ', positive.shape)