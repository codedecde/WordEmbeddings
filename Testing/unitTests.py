from TestClass import TestClass
import numpy as np
import cPickle as cp
import sys
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


class SubwordsClass(TestClass):
    def __init__(self, name, modelpath, bpe, subword2ix):
        # Load bpe models
        self.name = name
        self.bpe = bpe
        self.W_embed = np.load(modelpath)
        self.subword2ix = subword2ix
        super(SubwordsClass, self).__init__()

    def word2vec(self, w):
        w_seg = self.bpe.segment(w)
        w_embed = 0.
        for seg in w_seg:
            if seg in self.subword2ix:
                w_embed += self.W_embed[self.subword2ix[seg]]
        return w_embed


if __name__ == "__main__":
    # ============ Subword classifier ====================#
    sys.path.append('../')
    from Code.constants import *
    SUB_WORD_FILE = DATA_DIR + "BPE/vocab_subwords.txt"
    SUB_WORD_SEPERATOR = "@@"
    CODECS_FILE = DATA_DIR + "BPE/bpe_codecs.txt"
    from Code.bpe import BPE
    bpe = BPE(open(CODECS_FILE), separator=SUB_WORD_SEPERATOR)
    subword2ix = cp.load(open(SUBWORD_VOCAB_FILE))
    subword_tester = SubwordsClass("Subwords", "../Models/subword_vocab_matrix_with_syn_ant.npy", bpe, subword2ix)
    subword_tester.load_data()
    subword_tester.compute_stats()
    # ============= Word2vec Google =======================#
    glove_file = "../Baselines/Glove/glove.840B.300d.txt"
    print "Word2vec Embeddings"
    word2vec_tester = Word2vecEmbeddings("Word2vec", "../Models/Word2Vec/vectors_skipgram_300.bin")
    word2vec_tester.load_data()
    word2vec_tester.compute_stats()
    # ============= Word2vec Pytorch ======================#
    print "Word2vec Pytorch"
    word2vec_tester = EmbeddingFileTester("Word2vec_pytorch", "../Models/Embeddings_file.txt")
    word2vec_tester.load_data()
    word2vec_tester.compute_stats()
    # ============= Word2vec Pytorch with Syn =============#
    print "Word2vec Pytorch with syn constraints"
    word2vec_tester = EmbeddingFileTester("Word2vec_pytorch_with_syn", "../Models/embeddings_with_syn_info_epoch_5_no_lr_decay.txt")
    word2vec_tester.load_data()
    word2vec_tester.compute_stats()
    # ============= Fastext ===============================#
    print "Fast Text"
    fastext_tester = EmbeddingFileTester("Fast_Text", "../Models/vectors_fastext.txt")
    fastext_tester.load_data()
    fastext_tester.compute_stats()
    # ============= Counterfitted vectors =================#
    print "Counter Fitted Vectors ..."
    cfv_file = "../Baselines/Counter-fitted-vectors/counter-fitted-vectors.txt"
    glove_tester = EmbeddingFileTester(name="Counter Fitted Vectors", filepath=cfv_file)
    glove_tester.load_data()
    glove_tester.compute_stats()
    # ============= Glove Embeddings ======================#
    print 'Glove Embedding Vectors ...'
    glove_tester = EmbeddingFileTester(name="Glove Embedding Vectors", filepath=glove_file)
    #glove_tester = EmbeddingFileTester(name="Glove Embedding Vectors", filepath="../Models/glove.txt")
    glove_tester.load_data()
    glove_tester.compute_stats()
    # ======================================================#
