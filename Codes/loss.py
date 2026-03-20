import torch
import torch.nn as nn
import torch.nn.functional as F

class MyContrastiveLoss(nn.Module):

    def __init__(self, margin = 1.0):
        super(MyContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding_1, embedding_2, label):

        euclidean_distance = F.pairwise_distance(embedding_1, embedding_2)

        loss = label * euclidean_distance.pow(2) + (1 - label) * torch.clamp(self.margin - euclidean_distance, min = 0).pow(2)

        return loss.mean()
    
class MyTripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(MyTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, negative, positive):

        distance_negative = F.pairwise_distance(anchor, negative)
        distance_positive = F.pairwise_distance(anchor, positive)

        loss = torch.clamp(distance_positive - distance_negative + self.margin, min = 0)

        return loss.mean()

# if __name__ == '__main__':

#     embedding_1 = torch.randn(4, 128)
#     embedding_2 = torch.randn(4, 128)

#     labels = torch.tensor([0, 1, 0, 1], dtype = torch.float32)

#     contrastive = MyContrastiveLoss()
#     print('\ncontrasitive loss : ', contrastive(embedding_1, embedding_2, labels))

#     anchor = torch.randn(4, 128)
#     negative = torch.randn(4, 128)
#     positive = torch.randn(4, 128)

#     triplet = MyTripletLoss()
#     print('triplet loss : ', triplet(anchor, negative, positive))