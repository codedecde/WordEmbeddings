import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pdb


class Word2vec(nn.Module):
    def __init__(self, num_words, n_dim, sparse=False):
        super(Word2vec, self).__init__()
        self.num_words = num_words
        self.n_dim = n_dim

        self.embedding_i = nn.Embedding(num_words, n_dim, padding_idx=0, sparse=sparse)
        init_dim = np.sqrt(num_words)
        # self.embedding_i.weight = nn.Parameter(torch.Tensor(num_words, n_dim).uniform_(-1. / init_dim, 1. / init_dim))
        self.embedding_o = nn.Embedding(num_words, n_dim, padding_idx=0, sparse=sparse)
        # self.embedding_o.weight = nn.Parameter(torch.Tensor(num_words, n_dim).uniform_(-1. / init_dim, 1. / init_dim))

    def forward(self, w_ix, p_ix, neg_ix, syn_ix, ms_ix, ant_ix, ma_ix):
        """
        The forward pass
            :param w_ix: batch x 1: The words
            :param p_ix: batch x window_size: The positive examples
            :param neg_ix: batch x (window_size): The negative samples
            :param syn_ix: batch x n_syn: The synonyms
            :param ms_ix: batch x 1: The mask for synonyms
            :param ant_ix: batch x n_ant: The antonyms
            :param ma_ix: batch x 1: The mask for antomyms
        """
        # Handle the general case
        eps = 1e-10  # For numerical stability
        inp_embed = self.embedding_i(w_ix)  # batch x 1 x n_dim
        batch_size = inp_embed.size(0)
        p_embed = self.embedding_o(p_ix)  # batch x window x n_dim
        n_embed = self.embedding_o(neg_ix)  # batch x (window * neg_samples) x n_dim
        p_score = torch.sum(F.softplus((inp_embed * p_embed).sum(2) + eps, beta=-1)).neg()
        n_score = torch.sum(F.softplus((inp_embed * n_embed).sum(2).neg() + eps, beta=-1)).neg()
        loss = p_score + n_score
        # Now handle the synonyms and antonyms
        syn_embed = self.embedding_i(syn_ix)  # batch x n_syn x n_dim. Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(ms_ix * F.softplus((inp_embed * syn_embed).sum(2) + eps, beta=-1)).neg()
        loss += syn_score
        ant_embed = self.embedding_i(ant_ix)  # batch x n_ant x n_dim
        ant_score = torch.sum(ma_ix * F.softplus((inp_embed * ant_embed).sum(2).neg() + eps, beta=-1)).neg()
        loss += ant_score
        return loss / batch_size,\
            p_score / batch_size,\
            n_score / batch_size,\
            syn_score / batch_size,\
            ant_score / batch_size
