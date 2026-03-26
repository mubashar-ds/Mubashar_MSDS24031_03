import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import datasets

import argparse

def loading_data(mode, split = 'test'):

    embedding = np.load(f'../Embeddings/{mode}_test_embeddings.npy')
    labels = np.load(f'../Embeddings/{mode}_test_labels.npy')
    images = np.load(f'../Embeddings/{mode}_test_images.npy')

    return embedding, labels, images

# recall...

def recall_at_k(embeddings, labels, k = 1):
    
    labels = labels.flatten()

    correct = 0
    total_items = len(embeddings)

    for i in range(total_items):

        distance = np.linalg.norm(embeddings - embeddings[i], axis = 1)
        index = np.argsort(distance)[1:k+1]

        if k == 1:
            
            # strict nearest neighbor..
            if labels[i] == labels[index[0]]:
                correct += 1

        else:

            # top k match...
            if labels[i] in labels[index]:
                correct += 1

    return correct / total_items

# t sne..

def plotting_tsne(embeddings, labels, title):

    t_sne = TSNE(n_components = 2, perplexity = 30)

    embedding_2d = t_sne.fit_transform(embeddings)

    os.makedirs('../Graphs', exist_ok = True)

    plt.figure()
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c = labels, cmap = 'tab20', s = 5)
    plt.title(title)

    plt.savefig(f'../Graphs/{title}_tsne.png')
    plt.close()

# visualizing retreival...

def showing_retrieval(query_index, embeddings, images, labels, class_names, mode, k = 5):

    index = query_index

    dist = np.linalg.norm(embeddings - embeddings[index], axis = 1)
    nearest_neighbor_index = np.argsort(dist)[1:k+1]

    plt.figure(figsize =(10, 3))

    # query..

    plt.subplot(1, k+1, 1)
    plt.imshow(images[index])
    plt.title('Query')
    plt.axis('off')

    # neighbors...

    for i, j in enumerate(nearest_neighbor_index):

        plt.subplot(1, k+1, i+2)
        plt.imshow(images[j])

        color = 'green' if labels[j] == labels[index] else 'red'
        plt.title(class_names[labels[j]], color = color)

        plt.axis('off')

    plt.savefig(f'../Graphs/{mode}_retrieval_{index}.png')
    plt.close()

# runnign evaluation...

def evaluating(mode, data_path):

    dataset = datasets.ImageFolder(data_path)
    class_names = dataset.classes

    embeddings, labels, images = loading_data(mode, split = 'test')

    print(f'\n{mode} ---')

    r1 = recall_at_k(embeddings, labels, k = 1)
    print(f'\nRecall_at_1: {r1:.3f}')

    r5 = recall_at_k(embeddings, labels, k = 5)
    print(f'Recall_at_5: {r5:.3f}')

    plotting_tsne(embeddings, labels, mode)

    # 10 query visualizations..

    indexes = np.random.choice(len(embeddings), 10, replace = False)

    for i in indexes:   
        showing_retrieval(i, embeddings, images, labels, class_names, mode, k = 5)

    print('\nlabels shape :', labels.shape)
    print('sample labels :', labels[:5])
    print('type of labels[0] : ', type(labels[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--mode', type = str, required = True, choices=['contrastive', 'triplet', 'hard'])

    args = parser.parse_args()

    evaluating(args.mode, args.data_path)