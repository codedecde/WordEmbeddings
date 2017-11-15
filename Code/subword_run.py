from constants import *
import cPickle as cp
import torch.utils.data as ut
import torch
from utils import Progbar, getdata
from subword_model import subWord2vec
from collections import OrderedDict
import torch.optim as optim
from torch.autograd import Variable
import sys
from bpe import BPE
import numpy as np


# =========== CONSTANTS ==========================#
if DEBUG:
    TEXT = TINY_TEXT
use_cuda = torch.cuda.is_available()
data = filter(lambda x: len(x) > 1, open(TEXT).read().split(' '))
unigram_table = np.load(UNIGRAM_TABLE_FILE)
SUB_WORD_FREQ = 2
SUB_WORD_FILE = BASE_DIR + "BPE/vocab_subwords.txt"
SUB_WORD_SUFFIX = "@@"
CODECS_FILE = BASE_DIR + "BPE/bpe_codecs.txt"
MAX_SPLIT = 6
# =========== Load previous vocab ============#
word2ix = cp.load(open(VOCAB_FILE))
data = filter(lambda x: x in word2ix, data)
# =========== Build the vocab ====================#

counts = OrderedDict()
for line in open(SUB_WORD_FILE):
    line = line.strip().split()
    assert line[0] not in counts, "Duplicate %s found " % (line[0])
    counts[line[0]] = int(line[1])
counts = filter(lambda x: x[1] >= SUB_WORD_FREQ, sorted(counts.items(), key=lambda x: x[1], reverse=True))
subword2ix = {PAD_TOK: 0}
for w, c in counts:
    subword2ix[w] = len(subword2ix)


# =========== Now process synonyms and antonyms ===#
def add2dict(w1, w2, w_dict, word2ix):
    """Adds word to syn_dict or ant_dict
    """
    if w1 not in word2ix or w2 not in word2ix:
        return w_dict
    if w1 not in w_dict:
        w_dict[w1] = set()
    if w2 not in w_dict:
        w_dict[w2] = set()
    w_dict[w1].add(w2)
    w_dict[w2].add(w1)
    return w_dict


syn_dict = {}
ant_dict = {}

with open(PPDB_SYN_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        syn_dict = add2dict(line[0], line[1], syn_dict, word2ix)

with open(PPDB_ANT_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        ant_dict = add2dict(line[0], line[1], ant_dict, word2ix)

with open(WORDNET_ANT_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        ant_dict = add2dict(line[0], line[1], ant_dict, word2ix)
# =========== Preprocess data ====================#
'''
Preprocessing involves splitting data into byte pairs, ignoring OOV's and indexing tokens
Also involves adding synonyms and antonyms
'''
bpe = BPE(open(CODECS_FILE))


def index_data(data, window, bpe, subword2ix, syn_dict, ant_dict):
    """Indexes Data for data iterator
        :param data: [list of words]
        :param window: int : The positive window size / 2
        :param bpe: The byte pair encoder
        :param subword2ix: dictionary mapping subwords2ix
        :param syn_dict: dictionary from word to its synonyms
        :param ant_dict: dictionary from word to its antonyms
        :returns indexed_data: [(word), (context)]: where word: [MAX_SPLIT] and context: Window x MAX_SPLIT
        :returns syn_list: [[num_syn x MAX_SPLIT]]
        :returns ant_list: [[num_ant x MAX_SPLIT]]
    Note that this assumes the syn_dict and ant_dict have been cleaned appropreately
    """
    # Clean data:
    def index_segment(segmented_word, subword2ix):
        s = [subword2ix[e] for e in segmented_word][:MAX_SPLIT]
        s += [subword2ix[PAD_TOK] for _ in xrange(MAX_SPLIT - len(s))]
        return s
    clean = []
    for elem in data:
        s = bpe.segment(elem)
        if all(e in subword2ix for e in s):
            clean.append((elem, index_segment(s, subword2ix)))
    indexed_data = []
    syn_list = []
    ant_list = []
    words, indices = map(list, zip(*clean))
    bar = Progbar(len(clean) - window - window + 1)
    for ix in xrange(window, len(clean) - window):
        w = indices[ix]
        ctxt = indices[ix - window: ix] + indices[ix + 1: ix + window + 1]
        indexed_data.append((w, ctxt))
        syns = [index_segment(bpe.segment(_w), subword2ix) for _w in (syn_dict[words[ix]] if words[ix] in syn_dict else set())]
        syn_list.append(np.array(syns))
        ants = [index_segment(bpe.segment(_w), subword2ix) for _w in (ant_dict[words[ix]] if words[ix] in ant_dict else set())]
        ant_list.append(np.array(ants))
        bar.update(ix - window + 1)
    return indexed_data, syn_list, ant_list


# ============ Data Provider =========================#
class DataIterator(ut.Dataset):
    def __init__(self, **kwargs):
        self.words, self.ctxt = map(list, zip(*kwargs['indexed_data']))  # self.words: batch x T, self.ctxt: batch x window_size x T
        self.word2ix = kwargs['word2ix']
        self.ix2word = {self.word2ix[w]: w for w in self.word2ix}
        self.subword2ix = kwargs['subword2ix']
        self.unigram_table = kwargs['unigram_table']
        self.syn_list, self.ant_list = kwargs['syn_list'], kwargs['ant_list']
        self.n_neg = kwargs['n_neg']
        self.n_syn = kwargs['n_syn']
        self.n_ant = kwargs['n_ant']
        self.bpe = kwargs['bpe']

    def index_word(self, w):
        d = self.bpe.segment(w)
        for i in xrange(len(d)):
            assert d[i] in self.subword2ix, "Word %s not found " % (d[i])
            d[i] = self.subword2ix[d[i]]
        d = d[:MAX_SPLIT]
        d += [self.subword2ix[PAD_TOK] for _ in xrange(MAX_SPLIT - len(d))]
        return d

    def sample_negatives(self, number):
        words = [self.ix2word[ix] for ix in np.random.choice(self.unigram_table, replace=True, size=number)]
        n_ix = np.array([self.index_word(w) for w in words], dtype=int)
        return n_ix

    def sample(self, sample_table, number):
        if len(sample_table) == 0:
            m_ix = 0
            n_ix = [[self.subword2ix[PAD_TOK] for _ in xrange(MAX_SPLIT)] for _ in xrange(number)]
        else:
            m_ix = 1
            indices = np.random.choice(range(len(sample_table)), replace=True, size=number)
            n_ix = sample_table[indices]
        return n_ix, m_ix

    def __len__(self):
        return len(self.words)

    def __getitem__(self, ix):
        """
        """
        w_ix = self.words[ix]
        p_ix = self.ctxt[ix]
        n_ix = self.sample_negatives(self.n_neg)
        s_ix, ms_ix = self.sample(self.syn_list[ix], self.n_syn)
        a_ix, ma_ix = self.sample(self.ant_list[ix], self.n_ant)
        w_ix = torch.LongTensor(w_ix)
        p_ix = torch.LongTensor(p_ix)
        n_ix = torch.LongTensor(n_ix)
        s_ix = torch.LongTensor(s_ix)
        ms_ix = torch.FloatTensor([ms_ix])
        a_ix = torch.LongTensor(a_ix)
        ma_ix = torch.FloatTensor([ma_ix])
        return w_ix, p_ix, n_ix, s_ix, ms_ix, a_ix, ma_ix


window = 4
indexed_data, syn_list, ant_list = index_data(data, window=window,
                                              bpe=bpe, subword2ix=subword2ix,
                                              syn_dict=syn_dict, ant_dict=ant_dict)
neg_samples = 25
n_syn = 4
n_ant = 4
iterator = DataIterator(indexed_data=indexed_data, word2ix=word2ix,
                        subword2ix=subword2ix, unigram_table=unigram_table,
                        syn_list=syn_list, ant_list=ant_list, n_neg=neg_samples,
                        n_syn=n_syn, n_ant=n_ant, bpe=bpe)
BATCH_SIZE = 128
dataloader = ut.DataLoader(iterator, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)

N_EPOCHS = 5
lr = 0.025
bar = Progbar(N_EPOCHS)
sw2v = subWord2vec(len(subword2ix), 300, sparse=True)
if use_cuda:
    sw2v = sw2v.cuda()
optimizer = optim.Adagrad(sw2v.parameters(), lr=lr)
words_processed = 0.
for epoch in xrange(N_EPOCHS):
    n_batches = -(-len(iterator) // BATCH_SIZE)
    bar = Progbar(n_batches)
    print "\nEPOCH (%d/ %d)\n" % (epoch + 1, N_EPOCHS)
    for ix, batch in enumerate(dataloader):
        batch = map(lambda x: Variable(x), batch)
        if use_cuda:
            batch = map(lambda x: x.cuda(), batch)
        loss, p_score, n_score, s_score, a_score = sw2v(*batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        words_processed += BATCH_SIZE
        new_lr = lr * max(1e-4, 1. - (words_processed / (len(iterator) * N_EPOCHS)))
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = new_lr
        loss, p_score, n_score, s_score, a_score = map(lambda x: getdata(x).numpy()[0], [loss, p_score, n_score, s_score, a_score])
        bar.update(ix + 1, values=[('l', loss), ('p', p_score), ('n', n_score), ('s', s_score), ('a', a_score), ('lr', new_lr)])
weights = sw2v.embedding_i.weight
weights = weights.cpu() if use_cuda else weights
weights = weights.data.numpy()
save_file = BASE_DIR + "vocab_matrix_with_syn_ant.npy"
np.save(save_file, weights)
