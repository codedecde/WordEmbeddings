import numpy as np
import heapq


def cosine(x, y):
    eps = 1e-10
    return np.dot(x, y) / np.sqrt((np.dot(x, x) * np.dot(y, y)) + eps)


def get_nearest_k(word, vocab, vocab_matrix, k=4, return_score=False):
    k_nearest_neighbors = []
    vector_word = vocab_matrix[vocab[word]]
    for w in vocab:
        if w == word:
            continue
        dist = cosine(vector_word, vocab_matrix[vocab[w]])
        if len(k_nearest_neighbors) < k:
            heapq.heappush(k_nearest_neighbors, (dist, w))
        else:
            dist_min, _ = k_nearest_neighbors[0]
            if dist_min < dist:
                heapq.heappop(k_nearest_neighbors)
                heapq.heappush(k_nearest_neighbors, (dist, w))
    k_nearest_neighbors = [w for (d, w) in k_nearest_neighbors] if not return_score else k_nearest_neighbors
    return k_nearest_neighbors


def get_furthest_k(word, vocab, vocab_matrix, k=4, return_score=False):
    k_nearest_neighbors = []
    vector_word = vocab_matrix[vocab[word]]
    for w in vocab:
        if w == word:
            continue
        dist = -cosine(vector_word, vocab_matrix[vocab[w]])
        if len(k_nearest_neighbors) < k:
            heapq.heappush(k_nearest_neighbors, (dist, w))
        else:
            dist_min, _ = k_nearest_neighbors[0]
            if dist_min < dist:
                heapq.heappop(k_nearest_neighbors)
                heapq.heappush(k_nearest_neighbors, (dist, w))
    k_nearest_neighbors = [w for (d, w) in k_nearest_neighbors] if not return_score else k_nearest_neighbors
    return k_nearest_neighbors
