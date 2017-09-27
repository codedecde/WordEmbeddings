from TestClass import TestClass
import numpy as np


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
        # Load Superclass data
        super(EmbeddingFileTester, self).load_data()
        # Now read the Glove File
        with open(self.path) as f:
            for line in f:
                line = line.strip().split(' ')
                if self.in_test_vocab(line[0]):
                    self.word2vec_dict[line[0]] = np.array(map(lambda x: eval(x), line[1:]))


if __name__ == "__main__":
    cfv_file = "../Baselines/Counter-fitted-vectors/counter-fitted-vectors.txt"
    glove_file = "../Baselines/Glove/glove.840B.300d.txt"
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
