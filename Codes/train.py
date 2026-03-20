import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import MyEmbeddingNetwork
from dataset import MyContrastiveDataset, MyTripletDataset
from loss import MyContrastiveLoss, MyTripletLoss

import argparse

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

    loader = DataLoader(dataset, batch_size = 8, shuffle = True)

    for epoch in range(args.epochs):
        total_loss = 0

        for batch in loader:
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

        print(f'epoch {epoch+1}, loss : {total_loss}')

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type = str, required = True)
#     parser.add_argument('--mode', type = str, required = True, choices = ['contrastive', 'triplet'])
#     parser.add_argument('--epochs', type = int, default = 5)

#     args = parser.parse_args()
#     train(args)