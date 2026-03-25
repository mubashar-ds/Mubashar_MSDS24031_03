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

    def forward(self, anchor, positive, negative):

        distance_negative = F.pairwise_distance(anchor, negative)
        distance_positive = F.pairwise_distance(anchor, positive)

        loss = torch.clamp(distance_positive - distance_negative + self.margin, min = 0)

        return loss.mean()

def batch_hard_negative_mining(embeddings, labels, margin = 0.2):

    losses = []
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
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad = True)

    return torch.stack(losses).mean()

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
#     print('triplet loss : ', triplet(anchor, positive, negative))