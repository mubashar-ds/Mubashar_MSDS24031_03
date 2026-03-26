import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import argparse

from model import MyEmbeddingNetwork

def loading_model(model_path):

    model = MyEmbeddingNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def preprocessing_image(image_path):

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    image = Image.open(image_path).convert('RGB')

    return transform(image).unsqueeze(0)

def getting_embedding(model, image_path):

    image = preprocessing_image(image_path)

    with torch.no_grad():
        embedding = model(image)

    embedding = F.normalize(embedding, p = 2, dim = 1)

    return embedding

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--image_paths', type = str, nargs = '+', required =True)

    args = parser.parse_args()

    model = loading_model(args.model_path)

    embeddings = []

    for path in args.image_paths:

        embedding = getting_embedding(model, path)

        print(f'\nImage: {path}')
        print('embedding shape : ', embedding.shape)
        embeddings.append(embedding)

        print(embedding)
    
    if len(embeddings) >= 2:

        cos_similarity = F.cosine_similarity(embeddings[0], embeddings[1])
        print(f'\ncosine similarity : {cos_similarity.item():.3f}')