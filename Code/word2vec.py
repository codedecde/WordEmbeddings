import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np
import pdb
import io
from Lang import Vocab
from DataProcessing import iterator
from utils import Progbar
import torch.optim as optim
from IPython.core import debugger

# Some Torch constants
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class Word2Vec(nn.Module):
    def __init__(self, num_classes, embed_size, num_words):
        """
        :param num_classes: The number of possible classes.
        :param embed_size: EmbeddingLockup size
        :param num_words: The number of words you look at in an epoch
        """

        super(Word2Vec, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size).cuda() if use_cuda else nn.Embedding(self.num_classes, self.embed_size)

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size).cuda() if use_cuda else nn.Embedding(self.num_classes, self.embed_size)

        self.start_lr = 0.025
        self.optimizer = optim.SGD(self.parameters(), lr=self.start_lr, momentum=0.9)

        self.num_words = num_words
        self.words_processed = 0.

    def forward(self, input_tensor, target_tensor, neg_tensor):
        """
            :param target_tensor: batch x window size (LongTensor): The indices of target context windor
            :param input_tensor: batch x 1 (LongTensor): The indices of the input word
            :param neg_tensor: batch x neg_samples (LongTensor): The indices of the negatively sampled data
            :return loss (FloatTensor): The mean Negative Sampled Loss over batch
        """
        [batch_size, window_size] = target_tensor.size()
        [batch_size, neg_samples] = neg_tensor.size()
        input_embedding = self.in_embed(input_tensor)  # batch x n_dim
        target_embedding = self.out_embed(target_tensor)  # batch x Window x n_dim
        pos_score = torch.squeeze(torch.bmm(target_embedding, torch.unsqueeze(input_embedding, 2)), 2)  # batch x Window
        # pos_score = torch.sum(f.softplus(pos_score.neg()), -1) / window_size  # batch x 1
        pos_score = torch.sum(f.softplus(pos_score.neg()), -1)
        neg_embedding = self.out_embed(neg_tensor)  # batch x neg_samples x n_dim
        neg_score = torch.squeeze(torch.bmm(neg_embedding, torch.unsqueeze(input_embedding, 2)), 2)  # batch x neg_samples
        # neg_score = torch.sum(f.softplus(neg_score), -1) / neg_samples  # batch x 1
        neg_score = torch.sum(f.softplus(neg_score), -1)
        loss = torch.sum(pos_score + neg_score) / batch_size
        return loss

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
                self.words_processed += input_tensor.size(0)
                lr = self.start_lr * max(0.0001, 1. - (self.words_processed / (self.num_words * n_epochs + 1)))
                self.update_lr(lr)
                bar.update(step + 1, [('loss', loss_data), ('alpha', lr)])

    def save_embeddings(self, filename):
        data = self.in_embed.weight.data
        data = data.cpu().numpy() if use_cuda else data.numpy()
        np.save(open(filename, 'wb'), data)


if __name__ == "__main__":
    # Load the data
    filename = "../Data/text8"
    THRESHOLD = -1
    raw_data = io.open(filename, encoding='utf-8', mode='r', errors='replace').read(THRESHOLD).split(u' ')[1:]
    vocab_file = "../Models/Vocab_Mincount_10.pkl"
    vocab = Vocab()
    vocab.load_file(vocab_file)
    data = []
    for word in raw_data:
        if word in vocab.word2ix:
            data.append(word)
    batch_size = 256
    data_iterator = iterator(data, vocab, batch_size=batch_size)
    w2v = Word2Vec(num_classes=len(vocab), embed_size=300, num_words=len(data))
    steps_per_epoch = len(data) // batch_size if len(data) % batch_size == 0 else (len(data) // batch_size) + 1
    w2v.fit(data_iterator, n_epochs=5, steps_per_epoch=steps_per_epoch)
    # w2v.save_embeddings()
    model_save_file = "../Models/python_model.pkl"
    w2v.save_embeddings(model_save_file)
