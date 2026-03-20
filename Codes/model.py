import torch

from torchvision import models
from torchvision.models import ResNet50_Weights

import torch.nn as nn
import torch.nn.functional as F

class MyEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_vector = 128, backbone_freeze = True):
        super(MyEmbeddingNetwork, self).__init__()

        self.backbone = models.resnet50(weights = ResNet50_Weights.DEFAULT)
        input_size_fc_layer = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # freezing backbone as using cpu...
        if backbone_freeze:
            for layer_parameter in self.backbone.parameters():
                layer_parameter.requires_grad = False

        self.embedding = nn.Linear(input_size_fc_layer, embedding_vector)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p = 2, dim = 1)

        return x
    
# if __name__ == '__main__':

#     model = MyEmbeddingNetwork()
#     model.eval() 

#     dummy_input = torch.randn(1, 3, 32, 32)
#     output = model(dummy_input)

#     print('input shape : ', dummy_input.shape)
#     print('output shape : ', output.shape)
#     print('l2 norm : ', torch.norm(output, p=2, dim=1).item())