from Lang import Vocab
import heapq
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def cosine(x, y):
    eps = 0.000000001
    return np.dot(x, y) / np.sqrt((np.dot(x, x) * np.dot(y, y)) + eps)


def get_nearest_neighbors(word, vocab, vocab_matrix, k=4):
    k_nearest_neighbors = []
    vector_word = vocab_matrix[vocab[word]]
    for w in vocab.word2ix:
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
    k_nearest_neighbors = [w for (d, w) in k_nearest_neighbors]
    return k_nearest_neighbors


def get_nearest_neighbors_gensim(word, gensim_model, k=4):
    k_nearest_neighbors = []
    vector_word = gensim_model[word]
    for w in gensim_model.vocab.keys():
        if w == word:
            continue
        dist = cosine(vector_word, gensim_model[w])
        if len(k_nearest_neighbors) < k:
            heapq.heappush(k_nearest_neighbors, (dist, w))
        else:
            dist_min, _ = k_nearest_neighbors[0]
            if dist_min < dist:
                heapq.heappop(k_nearest_neighbors)
                heapq.heappush(k_nearest_neighbors, (dist, w))
    k_nearest_neighbors = [w for (d, w) in k_nearest_neighbors]
    return k_nearest_neighbors


if __name__ == "__main__":
    vocab_file = "../Models/Vocab_Mincount_10.pkl"
    vocab = Vocab()
    print "Loading vocab file from %s" % (vocab_file)
    vocab.load_file(vocab_file)

    gensim_filename = "../Models/Word2Vec/vectors_skipgram_300.bin"
    print "Loading model file from {}".format(gensim_filename)
    gensim_model = KeyedVectors.load_word2vec_format(gensim_filename, binary=True)

    vocab_matrix_filename = "../Models/word2vec_skipgram_neg_sample_25_window_8.npy"
    print "Loading model file from {}".format(vocab_matrix_filename)
    vocab_matrix = np.load(open(vocab_matrix_filename))

    words = ["king", "queen", "peasant"]
    k = 10
    for word in words:
        if word not in vocab.word2ix:
            print "word %s not found" % (word)
            continue
        nearest_neighbors = get_nearest_neighbors_gensim(word, gensim_model, k)
        print "Word : %s" % (word)
        print '\tGensim: '
        for neighbor in nearest_neighbors:
            print "\t\t%s (%.4f %.4f)" % (neighbor, cosine(gensim_model[word],
                                       gensim_model[neighbor]), cosine(vocab_matrix[vocab[word]],
                                       vocab_matrix[vocab[neighbor]]))
        nearest_neighbors = get_nearest_neighbors(word, vocab, vocab_matrix, k)
        print '\n\tPytorch: '
        for neighbor in nearest_neighbors:
            print "\t\t%s (%.4f %.4f)" % (neighbor, cosine(gensim_model[word],
                                       gensim_model[neighbor]), cosine(vocab_matrix[vocab[word]],
                                       vocab_matrix[vocab[neighbor]]))
        print ''

        # gold_set = set([w for w, _ in gensim_model.most_similar(word)])
        # assert all(w in gold_set for w in nearest_neighbors), "Error for word %s" % (word)
