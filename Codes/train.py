import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import MyEmbeddingNetwork
from dataset import MyContrastiveDataset, MyTripletDataset
from loss import MyContrastiveLoss, MyTripletLoss

import argparse

from torch.utils.data import random_split

import torch.nn.functional as F

from torchvision import transforms, datasets

import os

import matplotlib.pyplot as plt

def dataset_splits(dataset):

    training_size = int(0.7 * len(dataset))
    validation_size = int(0.15 * len(dataset))
    testing_size = len(dataset) - training_size - validation_size

    return random_split(dataset, [training_size, validation_size, testing_size])

def batch_hard_negative_mining(embeddings, labels, margin = 0.2):
    loss = 0.0
    batch_size = embeddings.size(0)

    for i in range(batch_size):
        anchor = embeddings[i]
        label_anchor = labels[i]

        distances = F.pairwise_distance(anchor.unsqueeze(0), embeddings)

        negative_mask = (labels != label_anchor)
        positive_mask = (labels == label_anchor)

        positive_mask[i] = False

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            continue

        hardest_negative = distances[negative_mask].min()
        hardest_positive = distances[positive_mask].max()

        loss += torch.clamp(hardest_positive - hardest_negative + margin, min = 0)

    return loss / batch_size

def train(args):

    os.makedirs('Graphs', exist_ok = True)

    os.makedirs('Saved_Models', exist_ok = True)

    os.makedirs('Training_Logs', exist_ok = True)

    device = 'cpu'

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    model = MyEmbeddingNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    if args.mode == 'contrastive':
        dataset = MyContrastiveDataset(args.data_path, transform)
        loss_function = MyContrastiveLoss()

    elif args.mode == 'triplet':
        dataset = MyTripletDataset(args.data_path, transform)
        loss_function = MyTripletLoss()

    elif args.mode == 'hard':
        from torchvision.datasets import ImageFolder
        dataset = datasets.ImageFolder(args.data_path, transform)

    training_dataset, validation_dataset, testing_dataset = dataset_splits(dataset)

    training_loader = DataLoader(training_dataset, batch_size = 8, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = 8)

    training_losses = []
    validation_losses = []

    # training loop...

    for epoch in range(args.epochs):

        model.train()
        total_loss = 0

        for batch in training_loader:
            optimizer.zero_grad()

            if args.mode == 'contrastive':
                image_1, image_2, label = batch
                loss = loss_function(model(image_1), model(image_2), label)

            elif args.mode == 'triplet':
                anchor, negative, positive = batch
                loss = loss_function(model(anchor), model(negative), model(positive))
            
            elif args.mode == 'hard':
                images, labels = batch
                embeddings = model(images)
                loss = batch_hard_negative_mining(embeddings, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_training_loss = total_loss / len(training_loader)
        training_losses.append(average_training_loss)

        print(f'epoch {epoch+1}, train loss : {total_loss}')

        # validation...

        model.eval()
        validation_loss = 0

        with torch.no_grad():

            for batch in validation_loader:

                if args.mode == 'contrastive':
                    image_1, image_2, label = batch
                    loss = loss_function(model(image_1), model(image_2), label)

                elif args.mode == 'triplet':
                    anchor, negative, positive = batch
                    loss = loss_function(model(anchor), model(negative), model(positive))
                
                elif args.mode == 'hard':
                    images, labels = batch
                    embeddings = model(images)
                    loss = batch_hard_negative_mining(embeddings, labels)

                validation_loss += loss.item()

        averageg_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(averageg_validation_loss)

        print(f'epoch {epoch+1}/{args.epochs}, validation loss: {averageg_validation_loss}')

        # model checkpoints saving...

        torch.save(model.state_dict(), f'Saved_Models/{args.mode}_epoch{epoch+1}.pth')

        # training logs..

        with open(f'Training_Logs/{args.mode}_log.txt', 'a') as f:
            f.write(f'{epoch+1},{average_training_loss},{averageg_validation_loss}\n')

    plt.plot(training_losses, label = 'Training')
    plt.plot(validation_losses, label='Validation')

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title(args.mode)

    plt.savefig(f'Graphs/{args.mode}_loss.png')

    plt.close()

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--data_path', type = str, required = True)
#     parser.add_argument('--mode', type = str, required = True, choices = ['Contrastive', 'Triplet', 'Hard'])

#     parser.add_argument('--epochs', type = int, default = 5)

#     args = parser.parse_args()

#     train(args)