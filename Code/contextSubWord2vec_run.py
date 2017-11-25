from constants import *
import cPickle as cp
import torch.utils.data as ut
import torch
from utils import Progbar, getdata, make_directory
from contextSubWord2vec_model import contextSubWord2vec
from collections import OrderedDict
import torch.optim as optim
import os
from torch.autograd import Variable
import argparse
import sys
import pdb
from bpe import BPE
import numpy as np


# =========== CONSTANTS ==========================#
use_cuda = torch.cuda.is_available()
data = filter(lambda x: len(x) > 1, open(TEXT).read().split(' '))
unigram_table = np.load(UNIGRAM_TABLE_FILE)
SUB_WORD_FREQ = 2
SUB_WORD_FILE = DATA_DIR + "BPE/vocab_subwords.txt"
SUB_WORD_SEPERATOR = "@@"
CODECS_FILE = DATA_DIR + "BPE/bpe_codecs.txt"
MAX_SPLIT = 4
SUBWORD_VOCAB_FILE = DATA_DIR + "BPE/" + ("subword2ix.dat" if DEBUG else "subword2ix_text8.dat")
BPE_DICT = DATA_DIR + "BPE/bpe_vocab.txt"
SAVE_PREFIX = BASE_DIR + "Models_Subwords/"
make_directory(SAVE_PREFIX)
SAVE_PREFIX += "BPE/"
make_directory(SAVE_PREFIX)
STOPWORDS_FILE = DATA_DIR + "stopwords.txt"
stopwords = set(map(lambda x: x.strip(), open(STOPWORDS_FILE).readlines()))
# =========== Load previous vocab ============#
word2ix = cp.load(open(VOCAB_FILE))
data = filter(lambda x: x in word2ix, data)
word2ix_nostop = OrderedDict()
for w in word2ix:
    if w not in stopwords:
        word2ix_nostop[w] = len(word2ix_nostop)
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

cp.dump(subword2ix, open(SUBWORD_VOCAB_FILE, 'wb'))


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
bpe = BPE(open(CODECS_FILE), separator=SUB_WORD_SEPERATOR)


def index_data(data, window, bpe, subword2ix, word2ix_nostop, syn_dict, ant_dict, stopwords):
    """Indexes Data for data iterator
        :param data: [list of words]
        :param window: int : The positive window size / 2
        :param bpe: The byte pair encoder
        :param subword2ix: dictionary mapping subwords2ix
        :param word2ix_nostop : dictionary mapping words to ix (doesn't contain stopwords)
        :param syn_dict: dictionary from word to its synonyms
        :param ant_dict: dictionary from word to its antonyms
        :param stopwords: The stopwords
        :returns indexed_data: [(word), (ctxt), (ctxt_nostop)]:
            where word: [MAX_SPLIT], context: Window x MAX_SPLIT, ctxt_nostop: window
        :returns syn_list: [[num_syn x MAX_SPLIT]]
        :returns ant_list: [[num_ant x MAX_SPLIT]]
    Note that this assumes the syn_dict and ant_dict have been cleaned appropreately
    """
    # Clean data:
    def get_ctxt_nostop(ix, words, window, word2ix_nostop, stopwords):
        left = ix - 1
        right = ix + 1
        words_left = []
        words_right = []
        while (left >= 0 or right < len(words)) and (len(words_left) + len(words_right) < (2 * window)):
            if left >= 0:
                w = words[left]
                if w not in stopwords:
                    words_left.append(word2ix_nostop[w])
                if len(words_left) + len(words_right) >= (2 * window):
                    break
            if right < len(words):
                w = words[right]
                if w not in stopwords:
                    words_right.append(word2ix_nostop[w])
            left -= 1
            right += 1
        words_left.reverse()
        ctxt_nostop = words_left + words_right
        ctxt_nostop += [word2ix_nostop[PAD_TOK] for _ in xrange((2 * window) - len(ctxt_nostop))]
        assert len(ctxt_nostop) == 2 * window, "Something is fishy"
        return ctxt_nostop

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
        ctxt_nostop = get_ctxt_nostop(ix, words, window, word2ix_nostop, stopwords)
        indexed_data.append((w, ctxt, ctxt_nostop))
        syns = [index_segment(bpe.segment(_w), subword2ix) for _w in (syn_dict[words[ix]] if words[ix] in syn_dict else set())]
        syn_list.append(np.array(syns))
        ants = [index_segment(bpe.segment(_w), subword2ix) for _w in (ant_dict[words[ix]] if words[ix] in ant_dict else set())]
        ant_list.append(np.array(ants))
        bar.update(ix - window + 1)
    return indexed_data, syn_list, ant_list


# ============ Data Provider =========================#
class DataIterator(ut.Dataset):
    def __init__(self, **kwargs):
        self.words, self.ctxt, self.ctxt_nostop = map(list, zip(*kwargs['indexed_data']))
        '''
            self.words: batch x T
            self.ctxt: batch x window_size x T
            self.ctxt_nostop : batch x window_size
        '''
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
        c_ix = self.ctxt_nostop[ix]
        n_ix = self.sample_negatives(self.n_neg)
        s_ix, ms_ix = self.sample(self.syn_list[ix], self.n_syn)
        a_ix, ma_ix = self.sample(self.ant_list[ix], self.n_ant)
        w_ix = torch.LongTensor(w_ix)
        p_ix = torch.LongTensor(p_ix)
        c_ix = torch.LongTensor(c_ix)
        n_ix = torch.LongTensor(n_ix)
        s_ix = torch.LongTensor(s_ix)
        ms_ix = torch.FloatTensor([ms_ix])
        a_ix = torch.LongTensor(a_ix)
        ma_ix = torch.FloatTensor([ma_ix])
        return w_ix, p_ix, c_ix, n_ix, s_ix, ms_ix, a_ix, ma_ix


def parse_args():
    def convert_boolean(args, field):
        if getattr(args, field).lower() in set(["false", "true"]):
            setattr(args, field, False if getattr(args, field).lower() == "false" else True)
        else:
            raise RuntimeError("value %s not valid for booleans" % (getattr(args, field)))

    parser = argparse.ArgumentParser(description="Word Embeddings")
    parser.add_argument("-w", "--window", help="window size", dest="window", default=4, type=int)
    parser.add_argument("-ns", "--neg_samples", help="window size", dest="neg_samples", default=25, type=int)
    parser.add_argument("-s", "--synonyms", help="Synonyms", dest="synonyms", default=4, type=int)
    parser.add_argument("-sg", "--scale_grad", help="Scale Gradients by frequency", dest="scale_grad", default="False", type=str)
    parser.add_argument("-cat", "--concatenate", help="Combine by concatenation", dest="cat", default="True", type=str)
    parser.add_argument("-a", "--antonyms", help="Antonyms", dest="antonyms", default=12, type=int)
    parser.add_argument("-b", "--batch", help="Batch Size", dest="batch", default=128, type=int)
    parser.add_argument("-e", "--epochs", help="Epochs", dest="epochs", default=5, type=int)
    parser.add_argument('-o', "--optimizer", help="Optimizer", dest='optimizer', default="Adagrad", type=str)
    args = parser.parse_args()
    convert_boolean(args, 'scale_grad')
    convert_boolean(args, 'cat')
    return args


args = parse_args()
window = args.window
indexed_data, syn_list, ant_list = index_data(data, window=window,
                                              bpe=bpe,
                                              subword2ix=subword2ix,
                                              word2ix_nostop=word2ix_nostop,
                                              syn_dict=syn_dict, ant_dict=ant_dict,
                                              stopwords=stopwords)
neg_samples = args.neg_samples
n_syn = args.synonyms
n_ant = args.antonyms
iterator = DataIterator(indexed_data=indexed_data, word2ix=word2ix,
                        subword2ix=subword2ix, unigram_table=unigram_table,
                        syn_list=syn_list, ant_list=ant_list, n_neg=neg_samples,
                        n_syn=n_syn, n_ant=n_ant, bpe=bpe)
BATCH_SIZE = args.batch
dataloader = ut.DataLoader(iterator, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)

N_EPOCHS = args.epochs
lr = 0.025
bar = Progbar(N_EPOCHS)
sw2v = contextSubWord2vec(len(subword2ix), 300, len(word2ix_nostop), sparse=False, scale_grad=args.scale_grad, cat=args.cat)
if use_cuda:
    sw2v = sw2v.cuda()

if hasattr(optim, args.optimizer):
    optimizer = getattr(optim, args.optimizer)(sw2v.parameters(), lr=lr)
else:
    print "Optimizer %s not found. Defaulting to Adagrad" % (args.optimizer)
    optimizer = optim.Adagrad(sw2v.parameters(), lr=lr)

words_processed = 0.


def save_model(model, filename):
    model_state_dict = model.state_dict()
    # Convert all cuda weights to cpu weights
    model_state_dict = {key: getdata(model_state_dict[key], False) for key in model_state_dict}
    save_dir = '/'.join(filename.split('/')[:-1])
    make_directory(save_dir)
    filename = filename if filename.endswith('.model') else filename + '.model'
    torch.save(model_state_dict, filename)


for epoch in xrange(N_EPOCHS):
    n_batches = -(-len(iterator) // BATCH_SIZE)
    bar = Progbar(n_batches)
    print "\nEPOCH (%d/ %d)\n" % (epoch + 1, N_EPOCHS)
    for ix, batch in enumerate(dataloader):
        batch = map(lambda x: Variable(x), batch)
        if use_cuda:
            batch = map(lambda x: x.cuda(), batch)
        loss, kl_loss, d_loss, p_score, n_score, s_score, a_score = sw2v(*batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        words_processed += BATCH_SIZE
        new_lr = lr * max(1e-4, 1. - (words_processed / (len(iterator) * N_EPOCHS)))
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = new_lr
        loss, kl_loss, d_loss, p_score, n_score, s_score, a_score = map(lambda x: getdata(x).numpy()[0], [loss, kl_loss, d_loss, p_score, n_score, s_score, a_score])
        bar.update(ix + 1, values=[('l', loss), ('kl', kl_loss), ('d', d_loss), ('p', p_score), ('n', n_score), ('s', s_score), ('a', a_score), ('lr', new_lr)])
    # Save model for persistance
    if epoch != N_EPOCHS - 1:
        save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}_partial".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        save_model(sw2v, save_file)
    else:
        partial_save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}_partial.model".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        os.remove(partial_save_file)
        save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        save_model(sw2v, save_file)
print ''
