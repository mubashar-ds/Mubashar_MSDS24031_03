import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import MyEmbeddingNetwork
from dataset import MyContrastiveDataset, MyTripletDataset, dataset_splits
from loss import MyContrastiveLoss, MyTripletLoss, batch_hard_negative_mining

import argparse

from torch.utils.data import random_split

import torch.nn.functional as F

from torchvision import transforms, datasets

import os

import matplotlib.pyplot as plt

from torch.utils.data import Subset
import random

def train(args):

    os.makedirs('../Graphs', exist_ok = True)

    os.makedirs('../Saved_Models', exist_ok = True)

    os.makedirs('../Training_Logs', exist_ok = True)

    device = 'cpu'

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    model = MyEmbeddingNetwork(backbone_freeze = True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)

    base_dataset = datasets.ImageFolder(args.data_path, transform)

    subset_size = 3000
    indices = random.sample(range(len(base_dataset)), subset_size)
    base_dataset = Subset(base_dataset, indices)

    training_ds, validation_ds, testing_ds = dataset_splits(base_dataset)

    if args.mode == 'contrastive':

        training_dataset = MyContrastiveDataset(training_ds)
        validation_dataset = MyContrastiveDataset(validation_ds)
        loss_function = MyContrastiveLoss()

    elif args.mode == 'triplet':

        training_dataset = MyTripletDataset(training_ds)
        validation_dataset = MyTripletDataset(validation_ds)
        loss_function = MyTripletLoss()

    elif args.mode == 'hard':

        training_dataset = training_ds
        validation_dataset = validation_ds

    training_loader = DataLoader(training_dataset, batch_size = 8, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = 8)

    training_losses = []
    validation_losses = []

    # training loop...

    best_validation_loss = float('inf')

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

        average_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(average_validation_loss)

        print(f'epoch {epoch+1}, validation loss: {average_validation_loss}')

        # best model saving...

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss

            torch.save(model.state_dict(), f'Saved_Models/{args.mode}_best.pth')
            print('best model saved...')

        # model checkpoints saving...

        torch.save(model.state_dict(), f'../Saved_Models/{args.mode}_epoch{epoch+1}.pth')

        # training logs..

        with open(f'../Training_Logs/{args.mode}_log.txt', 'a') as f:
            f.write(f'{epoch+1},{average_training_loss},{average_validation_loss}\n')

    plt.plot(training_losses, label = 'Training')
    plt.plot(validation_losses, label='Validation')

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title(args.mode)

    plt.savefig(f'../Graphs/{args.mode}_loss.png')

    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--mode', type = str, required = True, choices = ['contrastive', 'triplet', 'hard'])
    parser.add_argument('--epochs', type = int, default = 5)

    args = parser.parse_args()

    train(args)