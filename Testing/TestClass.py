'''
    An abstract class for generating all the tests
'''
import numpy as np
from scipy.stats import spearmanr
import pdb


def dot(x, y):
    return (np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))) + 1.


class TestClass(object):
    def __init__(self, data_dir="../Data/"):
        assert hasattr(self, 'name'), "%s has to be set by child class" % ("name")
        self.data_dir = data_dir
        self.SimLex_file = self.data_dir + "SimLex-999/SimLex-999.txt"
        self.SimVerb_file = self.data_dir + "SimVerb-3500/SimVerb-3500.txt"
        self.RareWords_file = self.data_dir + "RareWords/rw.txt"

    def word2vec(self, w):
        '''
            The child class implements this method. Return the word vector if it exists, else return None
        '''
        raise NotImplementedError

    def load_data(self):
        # The SimLex dataset
        self.all_words = set()
        self.SimLex = {"words": [], "labels": []}
        with open(self.SimLex_file, 'r') as f:
            # Small files. Can fit into memory
            data = f.readlines()
            for line in data[1:]:  # Ignore the first line
                line = line.strip().split('\t')
                self.SimLex["words"].append((line[0], line[1]))
                self.all_words.add(line[0])
                self.all_words.add(line[1])
                self.SimLex["labels"].append(eval(line[3]))
        # The SimVerb dataset
        self.SimVerb = {"words": [], "labels": []}
        with open(self.SimVerb_file, 'r') as f:
            # Small files. Can fit into memory
            data = f.readlines()
            for line in data:  # Don't ignore the first line
                line = line.strip().split('\t')
                self.SimVerb["words"].append((line[0], line[1]))
                self.all_words.add(line[0])
                self.all_words.add(line[1])
                self.SimVerb["labels"].append(eval(line[3]))
        # The RareWords dataset
        self.RareWords = {"words": [], "labels": []}
        with open(self.RareWords_file, 'r') as f:
            # Small files. Can fit into memort
            data = f.readlines()
            for line in data:
                line = line.strip().split('\t')
                self.RareWords["words"].append((line[0], line[1]))
                self.all_words.add(line[0])
                self.all_words.add(line[1])
                self.RareWords["labels"].append(eval(line[2]))

    def in_test_vocab(self, w):
        return True if w in self.all_words else False

    def compute_corr(self, attr_name):
        assert hasattr(self, attr_name), "%s not found" % (attr_name)
        predicted_scores = []
        max_score = max(getattr(self, attr_name)["labels"])
        num_unknown = 0.
        for ix, (w1, w2) in enumerate(getattr(self, attr_name)["words"]):
            vec_1 = self.word2vec(w1)
            vec_2 = self.word2vec(w2)
            if vec_1 is not None and vec_2 is not None:
                predicted_scores.append(dot(vec_1, vec_2))
            else:
                # Still don't know what to do about stuff that doesn't exist in the dictionary. Currently give the maximum possible penalty
                predicted_scores.append(max_score - getattr(self, attr_name)["labels"][ix])
                num_unknown += 1.
        corr, p_val = spearmanr(getattr(self, attr_name)["labels"], predicted_scores)
        return corr, p_val, (num_unknown / len(getattr(self, attr_name)["words"]))

    def compute_stats(self):
        # Compute for SimLex dataset
        corr_simlex, p_val_simlex, num_unknown_simlex = self.compute_corr("SimLex")
        # Compute for SimVerb dataset
        corr_simverb, p_val_simverb, num_unknown_simverb = self.compute_corr("SimVerb")
        corr_rarewords, p_val_rarewords, num_unknown_rarewords = self.compute_corr("RareWords")
        write_buf = "Model : %s\n" % (self.name)
        write_buf += "\tSimLex    Dataset : %.4f With %.4f unknown\n" % (corr_simlex, num_unknown_simlex)
        write_buf += "\tSimVerb   Dataset : %.4f With %.4f unknown\n" % (corr_simverb, num_unknown_simverb)
        write_buf += "\tRareWords Dataset : %.4f With %.4f unknown\n" % (corr_rarewords, num_unknown_rarewords)
        print write_buf


class DummyTestClass(TestClass):
    def __init__(self):
        self.name = "Random Model"
        super(DummyTestClass, self).__init__()

    def word2vec(self, word):
        return np.random.normal(size=(300, ))


if __name__ == "__main__":
    dummy_class = DummyTestClass()
    dummy_class.load_data()
    dummy_class.compute_stats()
