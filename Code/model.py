import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pdb

use_gpu = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor


class Word2vec(nn.Module):
    def __init__(self, num_words, n_dim, sparse=False):
        super(Word2vec, self).__init__()
        self.num_words = num_words
        self.n_dim = n_dim

        self.embedding_i = nn.Embedding(num_words, n_dim, padding_idx=0, sparse=sparse).type(FloatTensor)
        init_dim = np.sqrt(num_words)
        self.embedding_i.weight = nn.Parameter(torch.Tensor(num_words, n_dim).uniform_(-1. / init_dim, 1. / init_dim).type(FloatTensor))
        self.embedding_o = nn.Embedding(num_words, n_dim, padding_idx=0, sparse=sparse).type(FloatTensor)
        self.embedding_o.weight = nn.Parameter(torch.Tensor(num_words, n_dim).uniform_(-1. / init_dim, 1. / init_dim).type(FloatTensor))

    def forward(self, w_ix, p_ix, neg_ix, syn_ix, ant_ix):
        """
        The forward pass
            :param w_ix: batch x 1: The words
            :param p_ix: batch x window_size: The positive examples
            :param neg_ix: batch x (window_size): The negative samples
            :param syn_ix: batch x n_syn: The synonyms
            :param ant_ix: batch x n_ant: The antonyms
        """
        # Handle the general case
        w_ix = autograd.Variable(w_ix.type(LongTensor))
        p_ix = autograd.Variable(p_ix.type(LongTensor))
        neg_ix = autograd.Variable(neg_ix.type(LongTensor))
        eps = 1e-10  # For numerical stability
        inp_embed = self.embedding_i(w_ix)  # batch x 1 x n_dim
        batch_size = inp_embed.size(0)
        p_embed = self.embedding_o(p_ix)  # batch x window x n_dim
        n_embed = self.embedding_o(neg_ix)  # batch x (window * neg_samples) x n_dim
        p_score = torch.sum(F.logsigmoid((inp_embed * p_embed).sum(2) + eps)).neg()
        n_score = torch.sum(F.logsigmoid((inp_embed * n_embed).sum(2).neg() + eps)).neg()
        loss = p_score + n_score
        pos_score_data = p_score.cpu().data if use_gpu else p_score.data
        neg_score_data = n_score.cpu().data if use_gpu else n_score.data
        if np.isnan(pos_score_data.numpy()[0]) or np.isnan(neg_score_data.numpy()[0]):
            pdb.set_trace()

        # Now handle the synonyms and antonyms
        syn_ix = autograd.Variable(syn_ix.type(LongTensor))
        syn_embed = self.embedding_i(syn_ix)  # batch x n_syn x n_dim. Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(F.logsigmoid((inp_embed * syn_embed).sum(2) + eps)).neg()
        loss += syn_score
        ant_ix = autograd.Variable(ant_ix.type(LongTensor))
        ant_embed = self.embedding_i(ant_ix)  # batch x n_ant x n_dim
        ant_score = torch.sum(F.logsigmoid((inp_embed * ant_embed).sum(2).neg() + eps)).neg()
        loss += ant_score
        return loss / batch_size
