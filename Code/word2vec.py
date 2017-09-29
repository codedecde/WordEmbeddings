import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np
import pdb
import cPickle as cp
import io
from Lang import Vocab
from DataProcessing import iterator
from utils import Progbar
import torch.optim as optim

# Some Torch constants
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class Word2Vec(nn.Module):
    def __init__(self, num_classes, embed_size):
        """
        :param num_classes: The number of possible classes.
        :param embed_size: EmbeddingLockup size
        """

        super(Word2Vec, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size).type(FloatTensor)
        self.out_embed.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1).type(FloatTensor))

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1).type(FloatTensor))

        self.l2 = 0.0003
        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.l2)

    def forward(self, input_tensor, target_tensor, neg_tensor):
        """
            :param target_tensor: batch x window size (LongTensor): The indices of target context windor
            :param input_tensor: batch x 1 (LongTensor): The indices of the input word
            :param neg_tensor: batch x neg_samples (LongTensor): The indices of the negatively sampled data
            :return loss (FloatTensor): The mean Negative Sampled Loss over batch
        """
        [batch_size, window_size] = target_tensor.size()
        input_embedding = self.in_embed(input_tensor)  # batch x n_dim
        target_embedding = self.out_embed(target_tensor)  # batch x Window x n_dim
        pos_score = torch.squeeze(torch.bmm(target_embedding, torch.unsqueeze(input_embedding, 2)), 2)  # batch x Window
        neg_embedding = self.out_embed(neg_tensor)  # batch x neg_samples x n_dim
        neg_score = torch.squeeze(torch.bmm(neg_embedding, torch.unsqueeze(input_embedding, 2)), 2)  # batch x neg_samples
        loss = torch.sum(torch.sum(f.softplus(pos_score.neg()), -1) + torch.sum(f.softplus(neg_score), -1)) / batch_size
        return loss

    def fit(self, data_iterator, n_epochs, steps_per_epoch):
        """
            :param data_iterator: The iterator that generates data of form (input_tensor, output_tensor, negative_sampling)
            :param n_epochs: The number of epochs
            :param steps_per_epoch: Number of steps in a single epoch
        """
        for ix in xrange(n_epochs):
            print "\nEPOCH (%d/%d)" % (ix + 1, n_epochs)
            bar = Progbar(steps_per_epoch)
            for step in xrange(steps_per_epoch):
                input_tensor, output_tensor, neg_tensor = data_iterator.next()
                self.optimizer.zero_grad()
                loss = self.forward(Variable(input_tensor), Variable(output_tensor), Variable(neg_tensor))
                loss.backward()
                loss_data = loss.data.cpu().numpy()[0] if use_cuda else loss.data.numpy()[0]
                self.optimizer.step()
                bar.update(step + 1, [('loss', loss_data)])

    def save_embeddings(self, filename):
        data = self.in_embed.weight.data
        data = data.cpu().numpy() if use_cuda else data.numpy()
        cp.dump(data, open(filename, 'wb'))


if __name__ == "__main__":
    # Load the data
    filename = "../Data/text8"
    THRESHOLD = -1
    data = io.open(filename, encoding='utf-8', mode='r', errors='replace').read(THRESHOLD).split(u' ')[1:]
    vocab_file = "../Models/Vocab_Mincount_10.pkl"
    vocab = Vocab()
    vocab.load_file(vocab_file)
    batch_size = 256
    data_iterator = iterator(data, vocab, batch_size=batch_size)
    w2v = Word2Vec(num_classes=len(vocab), embed_size=300)
    steps_per_epoch = len(data) // batch_size if len(data) % batch_size == 0 else (len(data) // batch_size) + 1
    w2v.fit(data_iterator, n_epochs=1, steps_per_epoch=1)
    w2v.save_embeddings("../Models/python_model.pkl")
