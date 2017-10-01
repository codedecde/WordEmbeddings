import torch
import numpy as np
import pdb
import random
import io
from Lang import Vocab

# Some Torch constants
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def subsample(word, vocab):
    """
    Subsamples the word i.e with probability of subsampling rate drops the word
        :param word: The word to subsample
        :param vocab: The vocab (contains the subsample prob for each word)
        :return: True to keep the word, False otherwise
    """
    return True if random.random() < vocab.get_subsample_weights(word) else False


def negative_sample(vocab, batch_size, num_samples):
    """
    Generates negative samples
        :param vocab: The vocabulary (contains weights for negative sampling)
        :param batch_size: The batch size
        :param num_samples: Number of samples
        :return: batch_size x num_samples: Negative samples sampled from a multinomial
    """
    return torch.Tensor(np.random.choice(vocab.unigram_table, (batch_size, num_samples))).type(LongTensor)


def iterator(data, vocab, window_size=20, num_samples=5, batch_size=32):
    """
    The Data Iterator
        :param data: An list of words
        :param vocab: The vocabulary
        :param window_size: The context window size
        :param num_samples: Number of negative samples
        :param batch_size: The batch size
        :yield input_tensor: batch x 1 : The indexed input words
        :yield output_tensor: batch x window_size: The indexed context
        :yield neg_samples: batch x neg_samples: The indexed negative samples
    """
    def add_2_context(word, context_window, subsampled_unusued, vocab):
        if word in vocab.word2ix:
            # Ignore words not in dictionary
            if subsample(word, vocab):
                context_window.append(word)
            else:
                subsampled_unusued.append(word)
        else:
            print 'Warning: Cleaning not Done'
    stride = 1
    input_tensor = []
    output_tensor = []
    neg_samples = []
    while 1:
        for ix in xrange(0, len(data), stride):
            if data[ix] not in vocab.word2ix:
                print 'Warning: Cleaning not Done'
                continue
            # Generate the context window
            context = []
            subsampled_unusued = []
            l_ix = ix - 1
            r_ix = ix + 1
            while len(context) < window_size and (l_ix >= 0 or r_ix < len(data)):
                if l_ix >= 0:
                    add_2_context(data[l_ix], context, subsampled_unusued, vocab)
                    if len(context) == window_size:
                        break
                if r_ix < len(data):
                    add_2_context(data[r_ix], context, subsampled_unusued, vocab)
                    if len(context) == window_size:
                        break
                l_ix -= 1
                r_ix += 1
            # Fill out the context in case subsample returns false for everything
            jx = 0
            while len(context) < window_size and jx < len(subsampled_unusued):
                context.append(subsampled_unusued[jx])
                jx += 1
            context = np.array([vocab[w] for w in context])
            output_tensor.append(context)
            input_tensor.append(vocab[data[ix]])  # Vocab handles UNK tokens
            if len(input_tensor) == batch_size:
                input_tensor = torch.Tensor(np.array(input_tensor)).type(LongTensor)
                output_tensor = torch.Tensor(np.array(output_tensor)).type(LongTensor)
                # Generate the negative samples
                neg_samples = negative_sample(vocab, batch_size, num_samples)
                yield input_tensor, output_tensor, neg_samples
                input_tensor = []
                output_tensor = []
                neg_samples = []


def check_data_processing(input_tensor, output_tensor, neg_samples, vocab):
    """
    Tester function to visualize the indexing
        :param input_tensor: batch x 1: The indexed input word
        :param output_tensor: batch x window_size: The indexed context
        :param neg_samples: batch x neg_samples: The indexed negative samples
        :param vocab: The vocabulary
    """
    for inp, out, neg in zip(input_tensor, output_tensor, neg_samples):
        print 'Input Word : %s' % (vocab[inp])
        context = ' '.join(map(lambda x: vocab[x], out))
        print 'Context\n%s' % (context)
        negative_samples = ' '.join(map(lambda x: vocab[x], neg))
        print 'Negative Words\n%s' % (negative_samples)
        print ''


if __name__ == "__main__":
    vocab_file = "../Models/Vocab_Mincount_10.pkl"
    vocab = Vocab()
    vocab.load_file(vocab_file)
    ix = 0
    filename = "../Data/text8"
    THRESHOLD = -1
    data = io.open(filename, encoding='utf-8', mode='r', errors='replace').read(THRESHOLD).split(u' ')[1:]
    THRESHOLD = 5
    for input_tensor, output_tensor, neg_samples in iterator(data, vocab):
        check_data_processing(input_tensor, output_tensor, neg_samples, vocab)
        ix += 1
        if ix == THRESHOLD:
            break
