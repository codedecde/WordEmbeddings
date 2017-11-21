import numpy as np
import pdb
import sys
from gensim.models.keyedvectors import KeyedVectors
from utils import Progbar
import evaluate as E
import cPickle as cp


def find_closest_word(vector, word2vec):
    closest_val = None
    closest_word = None
    for w in word2vec:
        if closest_val is None or closest_val < E.cosine(word2vec[w], vector):
            closest_val = E.cosine(word2vec[w], vector)
            closest_word = w
    return closest_word


def compute_accuracy(method_metric, analogy_set):
    """
    Computes the word analogy accuracies for each method in the method_metric
        :param method_metric: A dictionary of form name : word2vec
        :param analogy_set: List of 4 tuples: word1, word2, word3, word4
        :return accuracies: A dictionary of form name: accuracy
    """
    accuracies = {name: 0. for name in method_metric}
    total = len(analogy_set)
    bar = Progbar(total)
    ix = 0
    for w1, w2, w3, w4 in analogy_set:
        ix += 1
        for method in accuracies:
            if any([w not in method_metric[method] for w in [w1, w2, w3, w4]]):
                continue
            w_vec = method_metric[method][w1] - method_metric[method][w2] + method_metric[method][w3]
            w = E.get_nearest_k_with_matrix(w_vec, method_metric[method], 5, False)
            if w4 in w:
                accuracies[method] += 1.
        bar.update(ix)
    accuracies = {name: accuracies[name] / total * 100 for name in accuracies}
    return accuracies


def load_word_vector(filename, wordset):
    word2vec = {}
    with open(filename) as f:
        for line in f:
            line = line.strip().split(' ')
            if line[0] in wordset:
                word2vec[line[0]] = np.array([float(x) for x in line[1:]])
    return word2vec


def load_w2v(fname, wordset):
    word_vectors = KeyedVectors.load_word2vec_format(fname, binary=True)
    word2vec = {}
    for w in wordset:
        if w in word_vectors.vocab:
            word2vec[w] = word_vectors[w]
    return word2vec


def load_sw2v(fname, wordset, bpe, subword2ix):
    word2vec = {}
    W_embed = np.load(fname)
    for word in wordset:
        w_seg = bpe.segment(word)
        w_embed = None
        for seg in w_seg:
            if seg in subword2ix:
                if w_embed is None:
                    w_embed = 0.
                w_embed += W_embed[subword2ix[seg]]
        if w_embed is not None:
            word2vec[word] = w_embed
    return word2vec


if __name__ == "__main__":
    analogy_file = "../Data/WordAnalogies/word_analogies.txt"
    analogies = []
    wordset = set()
    with open(analogy_file) as f:
        for line in f:
            if line.startswith(":"):
                continue
            line = [x.lower() for x in line.strip().split(' ')]
            for w in line:
                wordset.add(w)
            analogies.append(line)

    # =========== Loading Google Word2vec ====================================#
    word2vec = load_w2v("../Models/Word2Vec/vectors_skipgram_300.bin", wordset)
    # =========== Loading auxillary info for subword-model ===================#
    sys.path.append('../')
    from Code.constants import *
    SUB_WORD_FILE = DATA_DIR + "BPE/vocab_subwords.txt"
    SUB_WORD_SEPERATOR = "@@"
    CODECS_FILE = DATA_DIR + "BPE/bpe_codecs.txt"
    from Code.bpe import BPE
    bpe = BPE(open(CODECS_FILE), separator=SUB_WORD_SEPERATOR)
    subword2ix = cp.load(open(SUBWORD_VOCAB_FILE))
    # =========================================================================#
    methods = {"Word2vec": word2vec,
               "Word2vec Pytorch": load_word_vector("../Models/Embeddings_file.txt", wordset),
               "Word2vec with syn cons": load_word_vector("../Models/embeddings_with_syn_info_epoch_5_no_lr_decay.txt", wordset),
               "Fast Text": load_word_vector("../Models/fastext.vec", wordset),
               "Counter Fitted Vectors": load_word_vector("../Baselines/Counter-fitted-vectors/counter-fitted-vectors.txt", wordset),
               "Glove Embedding Vectors": load_word_vector("../Models/glove.txt", wordset),
               "Subword Embedding Vectors": load_sw2v("../Models/subword_vocab_matrix_with_syn_ant.npy", wordset, bpe, subword2ix)
               }
    accuracies = compute_accuracy(methods, analogies)
    print '\nPrinting the Analogy scores (@5)'
    for method in accuracies:
        print '\t%26s:\t%.4f' % (method, accuracies[method])
