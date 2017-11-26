import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb


class contextWord2vec(nn.Module):
    def __init__(self, num_words, n_dim, vocab_size, sparse=False, scale_grad=False, encoder_depth=2, mode="cat"):
        super(contextWord2vec, self).__init__()
        self.num_words = num_words
        self.n_dim = n_dim
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.in_dim = (n_dim // 2) if self.mode == "cat" else n_dim
        self.enc_embed_dim = n_dim if self.mode == "cat" else (2 * n_dim)
        for ix in xrange(encoder_depth):
            setattr(self, 'encoder_nn_{}'.format(ix), nn.Linear(self.enc_embed_dim, self.enc_embed_dim))
        self.enc_dim = (n_dim // 2) if self.mode == "cat" else n_dim
        self.encoder_mu = nn.Linear(self.enc_embed_dim, self.enc_dim)
        self.encoder_logvar = nn.Linear(self.enc_embed_dim, self.enc_dim)
        if self.mode == "tanh":
            self.project_embed = nn.Linear(2 * n_dim, n_dim)
        init_dim = np.sqrt(num_words)
        self.embedding_i = nn.Embedding(num_words, self.in_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_i = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_words, self.in_dim))
        e_i[0] = 0.
        self.embedding_i.weight = nn.Parameter(torch.Tensor(e_i))
        self.embedding_o = nn.Embedding(num_words, n_dim, padding_idx=0, sparse=sparse, scale_grad_by_freq=scale_grad)
        e_o = np.random.uniform(-1. / init_dim, 1. / init_dim, (num_words, n_dim))
        e_o[0] = 0.
        self.embedding_o.weight = nn.Parameter(torch.Tensor(e_o))
        self.embedding_c = nn.Embedding(vocab_size, n_dim)
        e_c = np.random.uniform(-1. / init_dim, 1. / init_dim, (vocab_size, self.enc_embed_dim))
        e_c[0] = 0.
        self.embedding_c.weight = nn.Parameter(torch.Tensor(e_c))

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

    def forward(self, w_ix, p_ix, c_ix, neg_ix, syn_ix, ms_ix, ant_ix, ma_ix):
        """
        The forward pass
            :param w_ix: batch x 1: The words
            :param p_ix: batch x window_size: The positive examples
            :param c_ix: batch x window_size: The context
            :param neg_ix: batch x (window_size): The negative samples
            :param syn_ix: batch x n_syn: The synonyms
            :param ms_ix: batch x 1: The mask for synonyms
            :param ant_ix: batch x n_ant: The antonyms
            :param ma_ix: batch x 1: The mask for antomyms
        """
        batch_size = w_ix.size(0)
        window_size = p_ix.size(1)
        neg_samples = neg_ix.size(1)
        # Encode Context
        mu, logvar = self.encode(c_ix)  # mu: batch x latent_dim, logvar: batch x latent_dim
        sigma = torch.exp(logvar / 2.)
        random_seed = Variable(torch.Tensor(np.random.multivariate_normal(np.zeros((sigma.size(1),)), np.eye(sigma.size(1), sigma.size(1)), batch_size)))
        if torch.cuda.is_available():
            random_seed = random_seed.cuda()
        z = mu + sigma * random_seed
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.mode in set(["cat", "sum", "tanh"]):
            kl_loss /= (window_size * neg_samples)
        else:
            kl_loss /= (neg_samples * window_size * self.enc_dim)

        # Decode
        eps = 1e-10  # For numerical stability
        partial_embed = self.embedding_i(w_ix)  # batch x 1 x (n_dim)

        if self.mode == "cat":
            inp_embed = torch.cat([z.unsqueeze(1), partial_embed], -1)
        elif self.mode == "sum":
            inp_embed = partial_embed + z.unsqueeze(1)
        elif self.mode == "prod":
            inp_embed = partial_embed * z.unsqueeze(1)
        elif self.mode == "tanh":
            inp_embed = torch.cat([z, partial_embed.squeeze(1)], -1)
            inp_embed = F.tanh(self.project_embed(inp_embed))
            inp_embed = inp_embed.unsqueeze(1)
        else:
            raise RuntimeError("Mode %s not recognized" % (self.mode))
        p_embed = self.embedding_o(p_ix)  # batch x window x n_dim
        n_embed = self.embedding_o(neg_ix)  # batch x (window * neg_samples) x n_dim
        p_score = torch.sum(F.softplus((inp_embed * p_embed).sum(2) + eps, beta=-1)).neg()
        n_score = torch.sum(F.softplus((inp_embed * n_embed).sum(2).neg() + eps, beta=-1)).neg()
        decoder_loss = p_score + n_score
        # Now handle the synonyms and antonyms
        syn_embed = self.embedding_i(syn_ix)  # batch x n_syn x (n_dim). Note that we want the constraints in the original vector space (Hence embedding_i)
        syn_score = torch.sum(ms_ix * F.softplus((partial_embed * syn_embed).sum(2) + eps, beta=-1)).neg()
        decoder_loss += syn_score
        ant_embed = self.embedding_i(ant_ix)  # batch x n_ant x (n_dim)
        ant_score = torch.sum(ma_ix * F.softplus((partial_embed * ant_embed).sum(2).neg() + eps, beta=-1)).neg()
        decoder_loss += ant_score
        loss = kl_loss + decoder_loss
        return loss / batch_size,\
            kl_loss / batch_size,\
            decoder_loss / batch_size,\
            p_score / batch_size,\
            n_score / batch_size,\
            syn_score / batch_size,\
            ant_score / batch_size


def load_context_word_model(state_dict, cat=False):
    if type(state_dict) == str:
        state_dict = torch.load(state_dict)
    num_words, n_dim = state_dict['embedding_o.weight'].size()
    vocab_size = state_dict['embedding_c.weight'].size(0)
    model = contextWord2vec(num_words, n_dim, vocab_size, cat=cat)
    model.load_state_dict(state_dict)
    return model


def word_embedding_given_context_and_word(model, context, word, word2ix_nostop, word2ix, numpy=True):
    ix_ctxt = []
    for w in context:
        if w in word2ix_nostop:
            ix_ctxt.append(word2ix_nostop[w])
    if len(ix_ctxt) > 0.:
        ix_ctxt = Variable(torch.LongTensor([ix_ctxt]))
        encoded_mu, encoded_logvar = model.encode(ix_ctxt)
    else:
        enc_dim = model.encoder_mu.out_features
        encoded_mu = Variable(torch.zeros(1, enc_dim))
    ix_word = word2ix[word]
    ix_word = Variable(torch.LongTensor([ix_word]))
    encoded_word = model.embedding_i(ix_word)
    if model.cat:
        embedding = torch.cat([encoded_mu, encoded_word], -1).view(-1, 1)
    else:
        upsampled = model.upsample(encoded_mu)
        embedding = (upsampled + encoded_word).view(-1, 1)
    if numpy:
        embedding = embedding.data.numpy().flatten()
        encoded_mu = encoded_mu.data.numpy().flatten()
        encoded_word = encoded_word.data.numpy().flatten()
    return embedding, encoded_mu, encoded_word
