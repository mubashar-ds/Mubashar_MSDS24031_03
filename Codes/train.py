import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import MyEmbeddingNetwork
from dataset import MyContrastiveDataset, MyTripletDataset
from loss import MyContrastiveLoss, MyTripletLoss

import argparse

from torch.utils.data import random_split

def dataset_splits(dataset):

    training_size = int(0.7 * len(dataset))
    validation_size = int(0.15 * len(dataset))
    testing_size = len(dataset) - training_size - validation_size

    return random_split(dataset, [training_size, validation_size, testing_size])


def train(args):

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

                validation_loss += loss.item()

        averageg_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(averageg_validation_loss)

        print(f'epoch {epoch+1}/{args.epochs}, validation loss: {averageg_validation_loss}')

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type = str, required = True)
#     parser.add_argument('--mode', type = str, required = True, choices = ['contrastive', 'triplet'])
#     parser.add_argument('--epochs', type = int, default = 5)

#     args = parser.parse_args()
#     train(args)