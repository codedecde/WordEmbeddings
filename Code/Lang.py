from __future__ import unicode_literals, print_function, division
from collections import Counter
import cPickle as cp
import numpy as np
import io
import pdb


VOCAB_SIZE = -1


class Vocab(object):
    def __init__(self, lowercase=True, tokenizer=None, min_count=10):
        self.tokenizer = tokenizer
        self.lowercase = lowercase  # To lowercase all words encountered
        self.unigram_table = []  # The unigram table for negative sampling
        self.subsample_weights = {}  # The weights for subsampling
        self.word2ix = {}  # The words to their index mapping
        self.ix2word = {}  # The index to the word
        self.min_count = min_count

    def get_subsample_weights(self, word):
        if word in self.subsample_weights:
            return self.subsample_weights[word]
        else:
            return 1.

    def generate_vocab(self, data):
        # Generate the counts
        word_counter = Counter()
        for word in data:
            if self.lowercase:
                word = word.lower()
            word_counter[word] += 1
        freq_table = word_counter.most_common(None if VOCAB_SIZE < 0 else VOCAB_SIZE)  # List [("word", frequency)]
        total_words = 0.
        for ix, (word, freq) in enumerate(freq_table):
            if freq < self.min_count:
                break
            self.word2ix[word] = ix
            self.ix2word[ix] = word
            total_words += freq
        neg_weights = np.zeros((len(self.word2ix), 1))
        for word in self.word2ix:
            z_w = word_counter[word] / total_words
            p_w = min(1., (0.001 / z_w) * (np.sqrt(z_w / 0.001) + 1))  # Also taken from Word2Vec paper
            self.subsample_weights[word] = p_w
            neg_weights[self.word2ix[word]] = (word_counter[word] ** (0.75))
        table_size = min(1000000, int(neg_weights.sum()))
        neg_weights = neg_weights / neg_weights.sum()
        neg_weights *= table_size
        self.unigram_table = []
        for ix in xrange(len(neg_weights)):
            for jx in xrange(int(neg_weights[ix])):
                self.unigram_table.append(ix)
                if len(self.unigram_table) == table_size:
                    break
            if len(self.unigram_table) == table_size:
                break

    def __getitem__(self, item):
        if type(item) == str or type(item) == unicode:
            # Encode the string to be unicode
            assert item in self.word2ix, "Word %s not in vocabulary" % (item)
            item = unicode(item)
            if self.lowercase:
                item = item.lower()
            return self.word2ix[item]
        else:
            assert item in self.ix2word, "Index %d not found" % (item)
            return self.ix2word[item]

    def __len__(self):
        assert len(self.ix2word) == len(self.word2ix), "Index not built using generate_vocab and add_word"
        return len(self.ix2word)  # We don't encode the <UNK> word

    def save_file(self, filename):
        cp.dump(self.__dict__, open(filename, 'wb'))

    def load_file(self, filename):
        self.__dict__ = cp.load(open(filename))


if __name__ == "__main__":
    data_dir = "/home/bass/DataDir/Embeddings/"
    data_file = data_dir + "Data/text8"
    THRESHOLD = -1
    data = io.open(data_file, encoding='utf-8', mode='r', errors='replace').read(THRESHOLD).split(u' ')[1:]
    vocab = Vocab()
    vocab.generate_vocab(data)
    save_filename = data_dir + "Data/Vocab_Mincount_10_with_unigram_table.pkl"
    vocab.save_file(save_filename)
