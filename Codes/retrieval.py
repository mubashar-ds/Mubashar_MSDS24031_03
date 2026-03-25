import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import datasets

def loading_data(prefix):

    images = np.load(f'../Embeddings/{prefix}_test_images.npy')
    labels = np.load(f'../Embeddings/{prefix}_test_labels.npy')
    embedding = np.load(f'../Embeddings/{prefix}_test_embeddings.npy')

    return embedding, labels, images

# recall...

def recall_at_k(embeddings, labels, k = 1):

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

# runnign evaluation...

def evaluating(prefix):

    embeddings, labels = loading_data(prefix)

    print(f'\n{prefix} ---')

    r1 = recall_at_k(embeddings, labels, k = 1)
    r5 = recall_at_k(embeddings, labels, k = 5)

    print(f'Recall_at_1: {r1:.3f}')
    print(f'Recall_at_5: {r5:.3f}')

    plotting_tsne(embeddings, labels, prefix)

if __name__ == '__main__':

    evaluating('contrastive')
    evaluating('triplet')
    evaluating('hard')