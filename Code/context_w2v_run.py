import torch.utils.data as ut
import os
import torch
import cPickle as cp
import pdb
import numpy as np
from collections import OrderedDict
from utils import Progbar, getdata, make_directory
from context_w2v_model import contextWord2vec
from torch.autograd import Variable
import argparse
import torch.optim as optim
from constants import *


use_cuda = torch.cuda.is_available()
data = filter(lambda x: len(x) > 1, open(TEXT).read().split(' '))
word2ix = cp.load(open(VOCAB_FILE))
unigram_table = np.load(UNIGRAM_TABLE_FILE)
data = filter(lambda x: x in word2ix, data)
STOPWORDS_FILE = DATA_DIR + "stopwords.txt"
stopwords = set(map(lambda x: x.strip(), open(STOPWORDS_FILE).readlines()))
word2ix_nostop = OrderedDict()
for w in word2ix:
    if w not in stopwords:
        word2ix_nostop[w] = len(word2ix_nostop)

SAVE_PREFIX = BASE_DIR + "Models/"
make_directory(SAVE_PREFIX)


def add2dict(w1, w2, w_dict, word2ix):
    if w1 not in word2ix or w2 not in word2ix:
        return w_dict
    w1 = word2ix[w1]
    w2 = word2ix[w2]
    if w1 not in w_dict:
        w_dict[w1] = set()
    if w2 not in w_dict:
        w_dict[w2] = set()
    w_dict[w1].add(w2)
    w_dict[w2].add(w1)
    return w_dict


syn_set = {}
ant_set = {}

with open(PPDB_SYN_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        syn_set = add2dict(line[0], line[1], syn_set, word2ix)

with open(PPDB_ANT_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        ant_set = add2dict(line[0], line[1], ant_set, word2ix)

with open(WORDNET_ANT_FILE) as f:
    for line in f:
        line = line.strip().split(' ')
        ant_set = add2dict(line[0], line[1], ant_set, word2ix)

# Convert the sets to lists
syn_set = {w: list(syn_set[w]) for w in syn_set}
ant_set = {w: list(ant_set[w]) for w in ant_set}


def generate_data(data, word2ix, word2ix_nostop, window_size, stopwords):
    """
    Takes in a sequence of words, and returns the indexed data, a list of (word, [2 * window])
        :param data: sequence of words
        :param word2ix: dictionary mapping words to indexes
        :param word2ix_nostop: dictionary without any stopwords
        :param window_size: Length of window
        :param stopwords: Set: Stopwords
        :return indexed_data: List of (word_ix, [2 * window], [2 * window]): The indexed word, the positive ctxt and the context
    """
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
    indexed_data = []
    for ix in xrange(window_size, len(data) - window_size):
        word_ix = word2ix[data[ix]]
        window = [word2ix[w] for w in data[ix - window_size: ix]] + [word2ix[w] for w in data[ix + 1: ix + window_size + 1]]
        ctxt_nostop = get_ctxt_nostop(ix, data, window_size, word2ix_nostop, stopwords)
        indexed_data.append((word_ix, window, ctxt_nostop))
    return indexed_data


class DataIterator(ut.Dataset):
    def __init__(self, unigram_table, indexed_data, neg_samples, syn_set, ant_set, n_syn, n_ant):
        self.indexed_data = indexed_data
        self.unigram_table = unigram_table.astype(int)
        self.neg_samples = neg_samples
        self.syn_set = syn_set
        self.ant_set = ant_set
        self.n_syn = n_syn
        self.n_ant = n_ant

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, idx):
        w_ix, p_ix, ctxt_ix = self.indexed_data[idx]
        n_ix = np.random.choice(self.unigram_table, replace=True, size=self.neg_samples)
        # Handle synonyms
        if w_ix in self.syn_set:
            syn_ix = np.random.choice(self.syn_set[w_ix], replace=True, size=self.n_syn)
            ms_ix = 1
        else:
            syn_ix = [0 for _ in xrange(self.n_syn)]  # 0 is a padding token
            ms_ix = 0
        if w_ix in self.ant_set:
            ant_ix = np.random.choice(self.ant_set[w_ix], replace=True, size=self.n_ant)
            ma_ix = 1
        else:
            ant_ix = [0 for _ in xrange(self.n_ant)]
            ma_ix = 0
        w_ix = torch.LongTensor([w_ix])
        n_ix = torch.LongTensor(n_ix)
        p_ix = torch.LongTensor(p_ix)
        c_ix = torch.LongTensor(ctxt_ix)
        syn_ix = torch.LongTensor(syn_ix)
        ms_ix = torch.FloatTensor([ms_ix])
        ant_ix = torch.LongTensor(ant_ix)
        ma_ix = torch.FloatTensor([ma_ix])
        return w_ix, p_ix, c_ix, n_ix, syn_ix, ms_ix, ant_ix, ma_ix


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
    parser.add_argument("-a", "--antonyms", help="Antonyms", dest="antonyms", default=4, type=int)
    parser.add_argument("-b", "--batch", help="Batch Size", dest="batch", default=128, type=int)
    parser.add_argument("-e", "--epochs", help="Epochs", dest="epochs", default=5, type=int)
    parser.add_argument('-o', "--optimizer", help="Optimizer", dest='optimizer', default="Adagrad", type=str)
    args = parser.parse_args()
    convert_boolean(args, 'scale_grad')
    convert_boolean(args, 'cat')
    return args


args = parse_args()
window = args.window
neg_samples = args.neg_samples
n_syn = args.synonyms
n_ant = args.antonyms

indexed_data = generate_data(data, word2ix, word2ix_nostop, window, stopwords)
iterator = DataIterator(unigram_table, indexed_data, neg_samples, syn_set, ant_set, n_syn, n_ant)
BATCH_SIZE = args.batch
dataloader = ut.DataLoader(iterator, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)

N_EPOCHS = args.epochs
# lr = 0.001
lr = 0.025
bar = Progbar(N_EPOCHS)
w2v = contextWord2vec(len(word2ix), 300, len(word2ix_nostop), sparse=False, scale_grad=args.scale_grad, cat=args.cat)

if use_cuda:
    w2v = w2v.cuda()

if hasattr(optim, args.optimizer):
    optimizer = getattr(optim, args.optimizer)(w2v.parameters(), lr=lr)
else:
    print "Optimizer %s not found. Defaulting to Adagrad" % (args.optimizer)
    optimizer = optim.Adagrad(w2v.parameters(), lr=lr)
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
    n_batches = len(iterator) // BATCH_SIZE if len(iterator) % BATCH_SIZE == 0 else (len(iterator) // BATCH_SIZE) + 1
    bar = Progbar(n_batches)
    print "\nEpoch (%d/ %d)\n" % (epoch + 1, N_EPOCHS)
    for ix, batch in enumerate(dataloader):
        batch = map(lambda x: Variable(x), batch)
        if use_cuda:
            batch = map(lambda x: x.cuda(), batch)
        loss, kl_loss, d_loss, p_score, n_score, s_score, a_score = w2v(*batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update the lr
        words_processed += BATCH_SIZE
        new_lr = lr * max(1e-4, 1. - (words_processed / (len(iterator) * N_EPOCHS)))
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = new_lr
        loss, kl_loss, d_loss, p_score, n_score, s_score, a_score = map(lambda x: getdata(x).numpy()[0], [loss, kl_loss, d_loss, p_score, n_score, s_score, a_score])
        bar.update(ix + 1, values=[('l', loss), ('kl', kl_loss), ('d', d_loss), ('p', p_score), ('n', n_score), ('s', s_score), ('a', a_score), ('lr', new_lr)])
    # Save model for persistence
    if epoch != N_EPOCHS - 1:
        save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}_partial".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        save_model(w2v, save_file)
    else:
        partial_save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}_partial.model".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        os.remove(partial_save_file)
        save_file = SAVE_PREFIX + "context_optim_{}_sg_{}_cat_{}_w_{}_ns_{}_s_{}_a_{}".format(args.optimizer, args.scale_grad, args.cat, args.window, args.neg_samples, args.synonyms, args.antonyms)
        save_model(w2v, save_file)
print ''
