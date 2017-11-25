import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb


class contextSubWord2vec(nn.Module):
    def __init__(self, num_units, n_dim, vocab_size, sparse=False, scale_grad=False, encoder_depth=2):
        super(contextSubWord2vec, self).__init__()
        self.num_units = num_units
        self.vocab_size = vocab_size
        self.encoder_depth = encoder_depth
        self.n_dim = n_dim

        for ix in xrange(encoder_depth):
            setattr(self, 'encoder_nn_{}'.format(ix), nn.Linear(n_dim, n_dim))
        self.encoder_mu = nn.Linear(n_dim, n_dim // 2)
        self.encoder_logvar = nn.Linear(n_dim, n_dim // 2)

        init_dim = np.sqrt(self.n_dim)
        self.embedding_i = nn.Embedding(num_units, (n_dim // 2), padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_i = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, (n_dim // 2)))
        e_i[0] = 0.
        self.embedding_i.weight = nn.Parameter(torch.Tensor(e_i))
        self.embedding_o = nn.Embedding(num_units, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_o = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_units, n_dim))
        e_o[0] = 0.
        self.embedding_o.weight = nn.Parameter(torch.Tensor(e_o))
        self.embedding_c = nn.Embedding(vocab_size, n_dim)
        e_c = np.random.uniform(-1. / init_dim, 1. / init_dim, (vocab_size, n_dim))
        e_c[0] = 0.
        self.embedding_c.weight = nn.Parameter(torch.Tensor(e_c))

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

    def encode(self, c_ix):
        """
        Encodes the context, and returns the mean and variance of the latent vector
            :param c_ix: batch x window_size: The context
            :returns mu: batch x latent_dim : The mean of the latent distribution
            :returns logvar: batch x latent_dim : log(sigma^2)
        """
        ctxt = torch.sum(self.embedding_c(c_ix), 1)  # batch x n_dim
        enc = ctxt
        for ix in xrange(self.encoder_depth):
            enc = F.tanh(getattr(self, 'encoder_nn_{}'.format(ix))(enc))
        mu = self.encoder_mu(enc)
        logvar = self.encoder_logvar(enc)
        return mu, logvar

    def forward(self, w_ix, p_ix, c_ix, n_ix, s_ix, ms_ix, a_ix, ma_ix):
        """
        The forward pass
            :param w_ix: batch x T: The sub-words
            :param p_ix: batch x window_size x T: The positive examples, split into sub units
            :param c_ix: batch x window_size: The context window
            :param n_ix: batch x neg_samples x T: The negative samples, split into sub units
            :param s_ix: batch x n_syn x T: The synonyms, split into sub units
            :param ms_ix: batch x 1: mask for synonyms
            :param a_ix: batch x n_ant x T: The antonyms, split into sub units
            :param ma_ix : batch x 1: mask for antonyms
        """
        batch_size = w_ix.size(0)
        window_size = c_ix.size(1)
        T = w_ix.size(-1)
        # Encode Context
        mu, logvar = self.encode(c_ix)  # mu: batch x latent_dim, logvar: batch x latent_dim
        sigma = torch.exp(logvar / 2.)
        random_seed = Variable(torch.Tensor(np.random.multivariate_normal(np.zeros((sigma.size(1),)), np.eye(sigma.size(1), sigma.size(1)), batch_size)))
        if torch.cuda.is_available():
            random_seed = random_seed.cuda()
        z = mu + sigma * random_seed
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= (window_size * T)
        # Decode
        eps = 1e-10  # For numerical stability
        partial_embed = torch.sum(self.embedding_i(w_ix), 1, keepdim=True)  # batch x 1 x (n_dim // 2)
        inp_embed = torch.cat([z.unsqueeze(1), partial_embed], -1)  # batch x 1 x n_dim
        p_embed = self.lookup(p_ix, self.embedding_o)  # batch x window x n_dim
        n_embed = self.lookup(n_ix, self.embedding_o)  # batch x neg_samples x n_dim
        p_score = torch.sum(F.softplus((inp_embed * p_embed).sum(2) + eps, beta=-1)).neg()
        n_score = torch.sum(F.softplus((inp_embed * n_embed).sum(2).neg() + eps, beta=-1)).neg()
        decoder_loss = p_score + n_score
        # Now handle the synonyms and antonyms
        syn_embed = self.lookup(s_ix, self.embedding_i)  # batch x n_syn x (n_dim // 2). Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(ms_ix * (F.softplus((partial_embed * syn_embed).sum(2) + eps, beta=-1))).neg()
        decoder_loss += syn_score

        ant_embed = self.lookup(a_ix, self.embedding_i)  # batch x n_ant x (n_dim // 2)
        ant_score = torch.sum(ma_ix * (F.softplus((partial_embed * ant_embed).sum(2).neg() + eps, beta=-1))).neg()
        decoder_loss += ant_score
        loss = kl_loss + decoder_loss
        return loss / batch_size,\
            kl_loss / batch_size,\
            decoder_loss / batch_size,\
            p_score / batch_size,\
            n_score / batch_size,\
            syn_score / batch_size,\
            ant_score / batch_size
