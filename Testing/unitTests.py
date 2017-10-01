from TestClass import TestClass
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class EmbeddingFileTester(TestClass):
    '''
        Runs the tester for the class of models which generate just the embedding file (like glove and Counter-fitted-vectors)
    '''
    def __init__(self, name, filepath):
        self.path = filepath
        self.name = name
        super(EmbeddingFileTester, self).__init__()
        self.word2vec_dict = {}

    def word2vec(self, w):
        return self.word2vec_dict[w] if w in self.word2vec_dict else None

    def load_data(self):
        # Load Parent data
        super(EmbeddingFileTester, self).load_data()
        # Now read the Glove File
        with open(self.path) as f:
            for line in f:
                line = line.strip().split(' ')
                if self.in_test_vocab(line[0]):
                    self.word2vec_dict[line[0]] = np.array(map(lambda x: eval(x), line[1:]))


class Word2vecEmbeddings(TestClass):
    '''
        Runs the tester for Word2vec bin models
    '''

    def __init__(self, name, modelpath):
        self.path = modelpath
        self.name = name
        super(Word2vecEmbeddings, self).__init__()

    def load_data(self):
        # Load parent data
        super(Word2vecEmbeddings, self).load_data()
        self.word_vectors = KeyedVectors.load_word2vec_format(self.path, binary=True)

    def word2vec(self, w):
        return self.word_vectors[w] if w in self.word_vectors else None


if __name__ == "__main__":
    cfv_file = "../Baselines/Counter-fitted-vectors/counter-fitted-vectors.txt"
    glove_file = "../Baselines/Glove/glove.840B.300d.txt"
    print "Word2vec Embeddings"
    word2vec_tester = Word2vecEmbeddings("Word2vec", "../Models/Word2Vec/vectors_skipgram_300.bin")
    word2vec_tester.load_data()
    word2vec_tester.compute_stats()
    print "Word2vec Pytorch"
    word2vec_tester = EmbeddingFileTester("Word2vec_pytorch", "../Models/word2vec_skipgram_neg_sample_25_window_8.txt")
    word2vec_tester.load_data()
    word2vec_tester.compute_stats()
    print "Counter Fitted Vectors ..."
    glove_tester = EmbeddingFileTester(name="Counter Fitted Vectors", filepath=cfv_file)
    print 'LOADING DATA ...'
    glove_tester.load_data()
    print 'DATA LOADED ....'
    glove_tester.compute_stats()
    print 'Glove Embedding Vectors ...'
    glove_tester = EmbeddingFileTester(name="Glove Embedding Vectors", filepath=glove_file)
    print 'LOADING DATA ...'
    glove_tester.load_data()
    print 'DATA LOADED ....'
    glove_tester.compute_stats()
