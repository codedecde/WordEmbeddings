import torch
import numpy as np
import pdb
import cPickle as cp
import random
import io
from Lang import Vocab

# Some Torch constants
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def subsample(word, vocab):
    return True if random.random() < vocab.get_subsample_weights(word) else False


def negative_sample(vocab, batch_size, num_samples):
    return torch.multinomial(vocab.weights, batch_size * num_samples, True).view(batch_size, num_samples).type(LongTensor)


def iterator(data, vocab, window_size=20, num_samples=5, batch_size=32):
    stride = 1
    left_stride = window_size // 2
    right_stride = window_size - left_stride
    input_tensor = []
    output_tensor = []
    neg_samples = []
    while 1:
        for ix in xrange(0, len(data), stride):
            # Generate the context window
            context = []
            subsampled_unusued = []
            if len(data) - ix < window_size:
                # Right boundary
                jx = ix + 1
                while jx < min(len(data), ix + right_stride + 1):
                    # Subsample for better context
                    if subsample(data[jx], vocab):
                        context.append(data[jx])
                    else:
                        subsampled_unusued.append(data[jx])
                    jx += 1
                jx = ix - 1
                while len(context) != window_size and jx >= 0:
                    # Subsample for better context
                    if subsample(data[jx], vocab):
                        context.append(data[jx])
                    else:
                        subsampled_unusued.append(data[jx])
                    jx -= 1
            else:
                jx = ix - 1
                while jx >= max(0, ix - left_stride - 1):
                    # Subsample for better context
                    if subsample(data[jx], vocab):
                        context.append(data[jx])
                    else:
                        subsampled_unusued.append(data[jx])
                    jx -= 1
                jx = ix + 1
                while len(context) != window_size and jx < len(data):
                    # Subsample for better context
                    if subsample(data[jx], vocab):
                        context.append(data[jx])
                    else:
                        subsampled_unusued.append(data[jx])
                    jx += 1
            # Fill out the context in case subsample returns false for everything
            jx = 0
            while len(context) != window_size and jx < len(subsampled_unusued):
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
