import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pdb

# class prod_reduce(autograd.Function):
#     def forward(self, inp, n_dim, keepdim=False):
#         """
#         Does the product reduce along axis 1
#         :param embed: batch x T x (n_dim * n_dim)
#         :param keepdim: bool: Keep the axis along which reduction happened
#         :param n_dim: int: The dimension. Note that n_dim = sqrt(embed.size(-1)), but is an unnecessary computation
#         :returns retval: batch x 1 x (n_dim * n_dim) if keepdim else batch x (n_dim * n_dim)
#         """
#         reshaped_inp = inp.view(inp.size(0), inp.size(1), n_dim, n_dim)
#         retval = None
#         for i in xrange(reshaped_inp.size(0)):
#             tmp = Variable(torch.Tensor(np.eye(self.n_dim)))
#             for j in xrange(reshaped_inp.size(1)):
#                 tmp = torch.mm(tmp, reshaped_inp[i, j])
#             retval = tmp.unsqueeze(0) if retval is None else torch.cat([retval, tmp.unsqueeze(0)])
#         retval = retval.view(retval.size(0), self.n_dim * self.n_dim)
#         if keepdim:
#             retval = retval.unsqueeze(1)

#         return retval



class subWord2mat(nn.Module):
    def __init__(self, num_units, n_dim, sparse=False, scale_grad=False):
        super(subWord2mat, self).__init__()
        self.num_units = num_units
        self.n_dim = n_dim
        init_dim = np.sqrt(self.n_dim)

        self.embedding_ia = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_ia = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_ia[0] = 0.
        self.embedding_ia.weight = nn.Parameter(torch.Tensor(e_ia))

        self.embedding_im = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_im = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_im[0] = 1.
        self.embedding_im.weight = nn.Parameter(torch.Tensor(e_im))

        self.embedding_oa = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_oa = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_oa[0] = 0.
        self.embedding_oa.weight = nn.Parameter(torch.Tensor(e_oa))

        self.embedding_om = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_om = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_om[0] = 1.
        self.embedding_om.weight = nn.Parameter(torch.Tensor(e_om))

    def lookup(self, ix, embedding):
        """
        This function does the lookup on the embedding matrix
        Unrolls the ix matrix, looks up the embedding matrix, sums over partial unit, and reshapes to required output
            :param ix: batch x W x T
            :param embedding: The embedding layer used as the lookup (embedding_i / embedding_o)
            :return embed: batch x W x n_embed
        """
        embed_a = getattr(self, embedding + 'a')(ix.view(-1, ix.size(-1)))  # (batch * W) x T x n_dim
        embed_m = getattr(self, embedding + 'm')(ix.view(-1, ix.size(-1)))  # (batch * W) x T x n_dim
        reduced_a = torch.sum(embed_a, 1, keepdim=False)  # (batch * W) x n_dim
        reduced_m = torch.prod(embed_m, 1, keepdim=False)  # (batch * W) x n_dim
        reduced_embed = torch.cat([reduced_a, reduced_m], -1)  # (batch * W) x (2 * n_dim)
        reduced_embed = reduced_embed.view(ix.size(0), ix.size(1), -1)  # batch x W x (2 * n_dim)
        return reduced_embed

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
        inp_a = torch.sum(self.embedding_ia(w_ix), 1, keepdim=True)  # batch x 1 x n_dim
        inp_m = torch.prod(self.embedding_im(w_ix), 1, keepdim=True)  # batch x 1 x n_dim
        inp_embed = torch.cat([inp_a, inp_m], -1)  # batch x 1 x (2 * n_dim)
        batch_size = inp_embed.size(0)

        p_embed = self.lookup(p_ix, 'embedding_o')  # batch x window x (n_dim * n_dim)
        n_embed = self.lookup(n_ix, 'embedding_o')  # batch x neg_samples x (n_dim * n_dim)
        p_score = torch.sum(F.softplus((inp_embed * p_embed).sum(2) + eps, beta=-1)).neg()
        n_score = torch.sum(F.softplus((inp_embed * n_embed).sum(2).neg() + eps, beta=-1)).neg()
        loss = p_score + n_score
        # Now handle the synonyms and antonyms
        syn_embed = self.lookup(s_ix, 'embedding_i')  # batch x n_syn x n_dim. Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(ms_ix * (F.softplus((inp_embed * syn_embed).sum(2) + eps, beta=-1))).neg()
        loss += syn_score

        ant_embed = self.lookup(a_ix, 'embedding_i')  # batch x n_ant x n_dim
        ant_score = torch.sum(ma_ix * (F.softplus((inp_embed * ant_embed).sum(2).neg() + eps, beta=-1))).neg()
        loss += ant_score
        return loss / batch_size,\
            p_score / batch_size,\
            n_score / batch_size,\
            syn_score / batch_size,\
            ant_score / batch_size
