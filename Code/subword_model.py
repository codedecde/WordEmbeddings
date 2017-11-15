import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class subWord2vec(nn.Module):
    def __init__(self, num_units, n_dim, sparse=False):
        super(subWord2vec, self).__init__()
        self.num_units = num_units
        self.n_dim = n_dim
        init_dim = np.sqrt(self.n_dim)
        self.embedding_i = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse)
        e_i = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_i[0] = 0.
        self.embedding_i.weight = nn.Parameter(torch.Tensor(e_i))
        self.embedding_o = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse)
        e_o = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_o[0] = 0.
        self.embedding_o.weight = nn.Parameter(torch.Tensor(e_o))

    def lookup(self, ix, embedding):
        """
        This function does the lookup on the embedding matrix
        Unrolls the ix matrix, looks up the embedding matrix, sums over partial unit, and reshapes to required output
            :param ix: batch x W x T
            :param embedding: The embedding layer used as the lookup (embedding_i / embedding_o)
            :return embed: batch x W x n_embed
        """
        embed = torch.sum(embedding(ix.view(-1, ix.size(-1))), -2).view(ix.size(0), ix.size(1), -1)  # batch x W x n_dim
        return embed

    def forward(self, w_ix, p_ix, n_ix, s_ix, ms_ix, a_ix, ma_ix):
        """
        The forward pass
            :param w_ix: batch x T: The sub-words
            :param p_ix: batch x window_size x T: The positive examples, split into sub units
            :param n_ix: batch x neg_samples x T: The negative samples, split into sub units
            :param s_ix: batch x n_syn x T: The synonyms, split into sub units
            :param ms_ix: batch x 1: mask for synonyms
            :param a_ix: batch x n_ant x T: The antonyms, split into sub units
            :param ma_ix : batch x 1: mask for antonyms
        """
        eps = 1e-10  # For numerical stability
        inp_embed = torch.sum(self.embedding_i(w_ix), 1, keepdim=True)  # batch x 1 x n_dim
        batch_size = inp_embed.size(0)

        p_embed = self.lookup(p_ix, self.embedding_o)  # batch x window x n_dim
        n_embed = self.lookup(n_ix, self.embedding_o)  # batch x neg_samples x n_dim
        p_score = torch.sum(F.softplus((inp_embed * p_embed).sum(2) + eps, beta=-1)).neg()
        n_score = torch.sum(F.softplus((inp_embed * n_embed).sum(2).neg() + eps, beta=-1)).neg()
        loss = p_score + n_score
        # Now handle the synonyms and antonyms
        syn_embed = self.lookup(s_ix, self.embedding_i)  # batch x n_syn x n_dim. Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(ms_ix * (F.softplus((inp_embed * syn_embed).sum(2) + eps, beta=-1))).neg()
        loss += syn_score

        ant_embed = self.lookup(a_ix, self.embedding_i)  # batch x n_ant x n_dim
        ant_score = torch.sum(ma_ix * (F.softplus((inp_embed * ant_embed).sum(2).neg() + eps, beta=-1))).neg()
        loss += ant_score
        return loss / batch_size,\
            p_score / batch_size,\
            n_score / batch_size,\
            syn_score / batch_size,\
            ant_score / batch_size
