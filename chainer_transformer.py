# -*- coding: utf-8 -*-
"""
Chainer port of [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
by Motoki Sato and Shun Kiyono
"""
import argparse
import collections
import copy
import math
import os
import random

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Sequential, cuda, Variable
from chainer import training
from chainer.training import extensions
from logzero import logger

import constant
from adam import VaswaniAdam
from dataset import DataProcessor
from resource import Resource
from updater import MultiProcessParallelUpdaterMod, MultiProcessParallelUpdaterMod_VAD

from order_samplers import ShuffleOrderSampler

def set_random_seed(seed, gpus):
    logger.info('Set random seed to {}'.format(seed))
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)

    # set CuPy random seed
    if len(gpus) == 1:
        if gpus[0] >= 0:
            chainer.cuda.get_device_from_id(gpus[0]).use()
            chainer.cuda.cupy.random.seed(seed)
    else:
        logger.info('CuPy random seed is not set, as it causes an error in Multi-GPU environment.')


def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, sentence_length, dimension),
    instead for array of ndim 2.
    """

    batch, length, units = x.shape
    e = func(x.reshape(batch * length, units))
    if not reconstruct_shape:
        return e
    else:
        return e.reshape(batch, length, -1)


### VAT
def get_normalized_vector(d, xp=None, shape=None):
    if shape is None:
        shape = tuple(range(1, len(d.shape)))
    d_norm = d
    if xp is not None:
        d_norm = d / (1e-12 + xp.max(xp.abs(d), shape, keepdims=True))
        d_norm = d_norm / xp.sqrt(1e-6 + xp.sum(d_norm ** 2, shape, keepdims=True))
    else:
        d_term = 1e-12 + F.max(F.absolute(d), shape, keepdims=True)
        d_norm = d / F.broadcast_to(d_term, d.shape)
        d_term = F.sqrt(1e-6 + F.sum(d ** 2, shape, keepdims=True))
        d_norm = d / F.broadcast_to(d_term, d.shape)
    return d_norm


def norm_vec_sentence_level(d, xp):
    # d         : (max_len, batchsize, emb_dim)
    # trans_d   : (batchsize, max_len, emb_dim)
    trans_d = xp.transpose(d, (1, 0, 2))
    norm_term = xp.linalg.norm(trans_d, axis=(1, 2), keepdims=True) + 1e-12
    trans_d = trans_d / norm_term
    d_sent_norm = xp.transpose(trans_d, (1, 0, 2))
    return d_sent_norm

def kl_loss(xp, p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))



class EncoderDecoder(Chain):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, share_embed=False,
                 criterion=None, use_vat=False, eps=1.0, perturbation_target=[0]):
        super(EncoderDecoder, self).__init__()

        with self.init_scope():
            self.encoder = Encoder(
                EncoderLayer(d_model, MultiHeadedAttention(h, d_model), PositionwiseFeedForward(d_model, d_ff, dropout),
                             dropout), N)
            self.decoder = Decoder(
                DecoderLayer(d_model, MultiHeadedAttention(h, d_model), MultiHeadedAttention(h, d_model),
                             PositionwiseFeedForward(d_model, d_ff, dropout), dropout), N)
            if share_embed:
                # source and target have equal size
                assert src_vocab == tgt_vocab, 'Source vocab size: {} != Target Vocab size: {}'.format(src_vocab,
                                                                                                       tgt_vocab)
                embed = Embeddings(d_model, src_vocab, eps=eps, use_vat=use_vat, perturbation_target=perturbation_target)
                self.embeddings_src = embed
                self.embeddings_trg = embed
                self.position_src = PositionalEncoding(d_model, dropout)
                self.position_trg = PositionalEncoding(d_model, dropout)

            else:
                self.embeddings_src = Embeddings(d_model, src_vocab, eps=eps, use_vat=use_vat, perturbation_target=perturbation_target)
                self.embeddings_trg = Embeddings(d_model, tgt_vocab, eps=eps, use_vat=use_vat, perturbation_target=perturbation_target)

                self.position_src = PositionalEncoding(d_model, dropout)
                self.position_trg = PositionalEncoding(d_model, dropout)

            self.generator = SharedOutputLayer(self.embeddings_trg.lut.W, bias=True)
            self.criterion = criterion
            self.perturbation_target = perturbation_target

        self.use_vat = use_vat

    def vat_adv_loss(self, batch, outputs, cleargrads_func, trg_y=None):
        # VAT first
        oVector = self.forward(batch, vat_step=0)
        xp = chainer.cuda.get_array_module(outputs)
        if self.use_vat == 1:
            # VAT
            loss_vat_first = kl_loss(xp, outputs.data, oVector)
        elif self.use_vat == 2:
            # Adv
            loss_vat_first = F.sum(F.softmax_cross_entropy(oVector, trg_y, normalize=False, ignore_label=constant.PAD_ID))
        elif self.use_vat == 3:
            # VAT + Adv
            loss_vat_first = kl_loss(xp, outputs.data, oVector)
            loss_vat_first += F.sum(F.softmax_cross_entropy(oVector, trg_y, normalize=False, ignore_label=constant.PAD_ID))


        cleargrads_func()
        loss_vat_first.backward()
        # VAT
        perturbation_enc = None
        perturbation_dec = None
        if 0 in self.perturbation_target:
            perturbation_enc = self.embeddings_src.perturbation_vars[0].grad
        if 1 in self.perturbation_target:
            perturbation_dec = self.embeddings_trg.perturbation_vars[1].grad

        oVector = self.forward(batch, perturbation_enc=perturbation_enc, perturbation_dec=perturbation_dec, vat_step=1)
        if self.use_vat == 1:
            # VAT
            loss = kl_loss(xp, outputs.data, oVector)
        elif self.use_vat == 2:
            # Adv
            loss = F.sum(F.softmax_cross_entropy(oVector, trg_y, normalize=False, ignore_label=constant.PAD_ID))
        elif self.use_vat == 3:
            # VAT + Adv
            loss = kl_loss(xp, outputs.data, oVector)
            loss += F.sum(F.softmax_cross_entropy(oVector, trg_y, normalize=False, ignore_label=constant.PAD_ID))


        return loss


    def forward(self, batch, perturbation_enc=None, perturbation_dec=None, vat_step=-1):
        h = self.encode(batch.src, batch.src_mask, perturbation=perturbation_enc, vat_step=vat_step)
        h = self.decode(h, batch.src_mask, batch.trg, batch.trg_mask, perturbation=perturbation_dec, vat_step=vat_step)
        outputs = self.generator(h)
        return outputs

    def __call__(self, batch, cleargrads_func=None):
        "Take in and process masked src and target sequences."
        # out = self.decode(self.encode(batch.src, batch.src_mask), batch.src_mask, batch.trg, batch.trg_mask)
        # x = self.generator(out)
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch.trg_y.reshape(-1)) / batch.n_tokens
        chainer.report({'loss': loss.data}, self)

        perp = self.xp.exp(loss.data)
        chainer.report({'perp': perp}, self)

        if self.use_vat and chainer.config.enable_backprop:
            if cleargrads_func is None:
                cleargrads_func = self.cleargrads
            loss_vat_or_adv = self.vat_adv_loss(batch, outputs, cleargrads_func=cleargrads_func, trg_y=batch.trg_y.reshape(-1))
            loss = loss + loss_vat_or_adv

        return loss

    def encode(self, src, src_mask, perturbation=None, vat_step=-1):
        # return self.encoder(self.src_embed(src, perturbation=perturbation, vat_step=vat_step), src_mask)
        h = self.embeddings_src(src, perturbation=perturbation, vat_step=vat_step, enc_or_dec=0)
        h = self.position_src(h)
        return self.encoder(h, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, perturbation=None, vat_step=-1):
        # return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # h = self.embeddings_trg(tgt)
        h = self.embeddings_trg(tgt, perturbation=perturbation, vat_step=vat_step, enc_or_dec=1)
        h = self.position_trg(h)
        return self.decoder(h, memory, src_mask, tgt_mask)


class SharedOutputLayer(chainer.Chain):
    def __init__(self, W, bias=True):
        super(SharedOutputLayer, self).__init__()
        self.W = W
        with self.init_scope():
            if bias is not None:
                self.add_param('b', (W.shape[0],), dtype='f')
                self.b.data[:] = 0.
            else:
                self.b = None

    def __call__(self, x):
        out = F.linear(x.reshape(-1, x.shape[-1]), self.W, self.b)
        return out


class Encoder(Chain):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layer_names = []
        with self.init_scope():
            for i in range(1, N + 1):
                name = 'l{}'.format(i)
                self.layer_names.append(name)
                setattr(self, name, copy.deepcopy(layer))
            self.norm = L.LayerNormalization(layer.size)

    def __call__(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layer_names:
            x = getattr(self, layer)(x, mask)
        return seq_func(self.norm, x)


class SublayerConnection(Chain):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        with self.init_scope():
            self.norm = L.LayerNormalization(size)
        self.dropout = dropout

    def __call__(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + F.dropout(sublayer(seq_func(self.norm, x)), self.dropout)


class EncoderLayer(Chain):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.sublayer1 = SublayerConnection(size, dropout)
            self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size

    def __call__(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)


class Decoder(Chain):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layer_names = []
        with self.init_scope():
            for i in range(1, N + 1):
                name = 'l{}'.format(i)
                self.layer_names.append(name)
                setattr(self, name, copy.deepcopy(layer))
            self.norm = L.LayerNormalization(layer.size)

    def __call__(self, x, memory, src_mask, tgt_mask):
        for layer in self.layer_names:
            x = getattr(self, layer)(x, memory, src_mask, tgt_mask)
        return seq_func(self.norm, x)


class DecoderLayer(Chain):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.size = size
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            self.sublayer1 = SublayerConnection(size, dropout)
            self.sublayer2 = SublayerConnection(size, dropout)
            self.sublayer3 = SublayerConnection(size, dropout)

    def __call__(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer3(x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape, dtype='i'), k=1)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = F.matmul(query, F.swapaxes(key, -2, -1)) / math.sqrt(d_k)
    if mask is not None:
        xp = chainer.cuda.get_array_module(key)
        minfs = xp.full(scores.shape, -1e9, 'f')
        scores = F.where(F.broadcast_to(mask, scores.shape), scores, minfs)
    p_attn = F.softmax(scores, axis=key.ndim - 1)
    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout)
    return F.matmul(p_attn, value), p_attn


class MultiHeadedAttention(chainer.ChainList):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.dropout = dropout

        linears = [
            ConvolutionSentence(d_model, d_model),
            ConvolutionSentence(d_model, d_model),
            ConvolutionSentence(d_model, d_model),
            ConvolutionSentence(d_model, d_model)
        ]
        super(MultiHeadedAttention, self).__init__(*linears)
        self.attn = None

    def __call__(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask[:, None]
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [F.swapaxes(F.swapaxes(linear(F.swapaxes(x, -2, -1)), -1, -2).reshape(nbatches, -1, self.h, self.d_k), 1, 2)
             for linear, x in
             zip(self, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = F.swapaxes(x, 1, 2).reshape(nbatches, -1, self.h * self.d_k)
        return F.swapaxes(self[-1](F.swapaxes(x, -2, -1)), -2, -1)


class PositionwiseFeedForward(Chain):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        with self.init_scope():
            self.w_1 = ConvolutionSentence(d_model, d_ff)
            self.w_2 = ConvolutionSentence(d_ff, d_model)
        self.dropout = dropout

    def __call__(self, x):
        x = self.w_2(F.dropout(F.relu(self.w_1(x.transpose(0, 2, 1))), self.dropout))
        return x.transpose(0, 2, 1)


class ConvolutionSentence(L.Convolution2D):
    """ Position-wise Linear Layer for Sentence Block

    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.

    """

    def __init__(self, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).

        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).

        """
        x = F.expand_dims(x, axis=3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y


class Embeddings(Chain):
    def __init__(self, d_model, vocab, eps=1.0, use_vat=0, perturbation_target=None):
        super(Embeddings, self).__init__()
        with self.init_scope():
            self.lut = L.EmbedID(vocab, d_model)
        self.d_model = d_model
        self.scale_emb = d_model ** 0.5
        self.eps = eps
        self.use_vat = use_vat
        self.perturbation_target = perturbation_target
        # if 0 in perturbation_target:
        #     self.use_vat_enc = True
        # if 1 in perturbation_target:
        #     self.use_vat_dec = True

        self.perturbation_vars = [None for _ in range(2)] # enc, dec

    def use_vat_cal(self, enc_or_dec):
        if self.perturbation_target is None:
            return False
        if enc_or_dec in self.perturbation_target:
            return self.use_vat
        return False


    def __call__(self, x, perturbation=None, vat_step=-1, enc_or_dec=0):
        use_vat = self.use_vat_cal(enc_or_dec)
        emb = self.lut(x)
        xp = chainer.cuda.get_array_module(x)
        if use_vat == 0:
            return emb * self.scale_emb
        if vat_step == -1:
            return emb * self.scale_emb
        elif vat_step == 0:
            if use_vat == 2:
                # Adv
                perturbation = xp.zeros(emb.shape, dtype='f')
            elif use_vat == 3:
                # VAT + Adv
                perturbation = xp.random.normal(size=emb.shape, dtype='f')
            else:
                # VAT
                perturbation = xp.random.normal(size=emb.shape, dtype='f')
            self.perturbation_var = Variable(perturbation)
            perturbation = self.perturbation_var
            # self.perturbation_vars[int(perturbation.data.device)] = self.perturbation_var
            self.perturbation_vars[enc_or_dec] = self.perturbation_var
            # Normalize
            perturbation = get_normalized_vector(perturbation)

        elif vat_step == 1:
            # Sentence-level Normalization
            perturbation = self.eps * norm_vec_sentence_level(perturbation, xp)

        # print('vat_step:', vat_step, 'perturbation:', perturbation.shape, ' emb:', emb.shape)
        return (perturbation + emb) * self.scale_emb

class PositionalEncoding(Chain):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout

        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(self.xp.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None, :].astype(np.float32)
        with self.init_scope():
            self.pe = chainer.Parameter(initializer=pe, shape=pe.shape)
        # Positional Encodingは更新しない
        self.pe._requires_grad = False

    def __call__(self, x):
        pos_embed = self.pe[:, :x.shape[1]]
        x = x + F.broadcast_to(pos_embed, x.shape)
        return F.dropout(x, self.dropout)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, share_embed=False, smoothing=0.1, use_vat=0, eps=1.0, perturbation_target=[0]):
    "Helper: Construct a model from hyperparameters."
    if smoothing > 0.0:
        criterion = LabelSmoothing(size=tgt_vocab, padding_idx=constant.PAD_ID, smoothing=smoothing)
    else:
        criterion = CrossEntropy(padding_idx=constant.PAD_ID)
    model = EncoderDecoder(src_vocab, tgt_vocab, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,
                           share_embed=share_embed, criterion=criterion, use_vat=use_vat, eps=eps, perturbation_target=perturbation_target)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    glorot_initializer = chainer.initializers.GlorotUniform()
    for name, param in model.namedparams():
        # Do not initialize positional encoding (PE) here
        # --> PE is initialized by its own __init__ method
        if param.ndim > 1 and 'pe' not in name:
            param.copydata(chainer.Parameter(
                glorot_initializer, param.shape))
    return model


class Batch(object):
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = np.stack(self.pad_instance(src, pad=pad)).astype('i')
        self.src_mask = (self.src != pad)[:, None, :]

        if trg is not None:
            trg = np.stack(self.pad_instance(trg, pad=pad)).astype('i')
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.n_tokens = (self.trg_y != pad).sum()
        else:
            self.trg = None
            self.trg_y = None
            self.trg_mask = None

    def to_gpu(self, device_id):
        self.src = cuda.to_gpu(self.src, device_id)
        self.src_mask = cuda.to_gpu(self.src_mask, device_id)
        if self.trg is not None:
            self.trg = cuda.to_gpu(self.trg, device_id)
            self.trg_mask = cuda.to_gpu(self.trg_mask, device_id)
            self.trg_y = cuda.to_gpu(self.trg_y, device_id)

    @staticmethod
    def pad_instance(lis, pad):
        max_len = max(len(x) for x in lis)
        out = [np.concatenate([x, [pad] * (max_len - len(x))]) for x in lis]
        return out

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad)[:, None, :]
        tgt_mask = tgt_mask & subsequent_mask(tgt.shape[-1])
        return tgt_mask

def kl_divergence(xs, target):
    return F.sum(target * (F.log(target + 1e-8) - F.log_softmax(xs)))


class LabelSmoothing(Chain):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def __call__(self, x, target):
        xp = chainer.cuda.get_array_module(x)
        assert x.shape[1] == self.size
        true_dist = xp.full(x.shape, self.smoothing / (self.size - 2)).astype('float32')
        true_dist[xp.arange(target.shape[0]), target] = self.confidence
        true_dist[:, self.padding_idx] = 0
        mask = self.xp.nonzero(target == self.padding_idx)
        if mask[0].size != 0:
            true_dist[mask] = 0.0
        self.true_dist = true_dist
        return kl_divergence(x, true_dist)


class CrossEntropy(Chain):
    def __init__(self, padding_idx=-1):
        super(CrossEntropy, self).__init__()
        self.padding_idx = padding_idx

    def __call__(self, xs, t):
        return F.sum(F.softmax_cross_entropy(xs, t, ignore_label=self.padding_idx, reduce='no'))


def get_topk(output, k=20):
    batchsize, n_out = output.shape
    xp = cuda.get_array_module(output)
    argsort = xp.argsort(output, axis=1)
    argtopk = argsort[:, ::-1][:, :k]
    assert (argtopk.shape == (batchsize, k)), (argtopk.shape, (batchsize, k))
    topk_score = output.take(
        argtopk + xp.arange(batchsize)[:, None] * n_out)
    return argtopk, topk_score


def update_beam_state(outs, total_score, topk, topk_score, eos_id):
    xp = cuda.get_array_module(outs)
    full = outs.shape[0]
    prev_full, k = topk.shape
    batch = full // k
    prev_k = prev_full // batch
    assert (prev_k in [1, k])

    if total_score is None:
        total_score = topk_score
    else:
        is_end = xp.max(outs == eos_id, axis=1)  # if candidate in the beam already outputs EOS token.
        is_end = xp.broadcast_to(is_end[:, None], topk_score.shape)
        bias = xp.zeros_like(topk_score, np.float32)
        bias[:, 1:] = -10000.  # remove finished candidates except for a consequence
        total_score = xp.where(
            is_end,
            total_score[:, None] + bias,
            total_score[:, None] + topk_score)
        assert (xp.all(total_score < 0.))
        topk = xp.where(is_end, eos_id, topk)  # this is not required
    total_score = total_score.reshape((prev_full // prev_k, prev_k * k))
    argtopk, total_topk_score = get_topk(total_score, k=k)
    assert (argtopk.shape == (prev_full // prev_k, k))
    assert (total_topk_score.shape == (prev_full // prev_k, k))
    total_topk = topk.take(
        argtopk + xp.arange(prev_full // prev_k)[:, None] * prev_k * k)
    total_topk = total_topk.reshape((full,))
    total_topk_score = total_topk_score.reshape((full,))

    argtopk = argtopk // k + \
              xp.arange(prev_full // prev_k)[:, None] * prev_k
    argtopk = argtopk.reshape((full,)).tolist()

    outs = xp.stack([outs[i] for i in argtopk], axis=0)
    outs = xp.concatenate([outs, total_topk[:, None]], axis=1).astype(np.int32)
    return outs, total_topk_score


def finish_beam(outs, total_score, batchsize, eos_id, length_penalty):
    k = outs.shape[0] // batchsize
    result_batch = collections.defaultdict(
        lambda: {'outs': [], 'score': -1e8})
    for i in range(batchsize):
        for j in range(k):
            score = total_score[i * k + j]
            out = outs[i * k + j].tolist()
            if eos_id in out:
                out = out[:out.index(eos_id)]
                score /= len(out) ** length_penalty
            if result_batch[i]['score'] < score:
                result_batch[i] = {'outs': out, 'score': score}

    result_batch = [
        result for i, result in
        sorted(result_batch.items(), key=lambda x: x[0])]
    return result_batch


def beam_decode(model, src, src_mask, max_len, start_symbol, k, length_penalty=0.6):
    batchsize = src.shape[0]
    xp = cuda.get_array_module(src)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        memory = model.encode(src, src_mask)
        ys = xp.full((memory.shape[0], 1), start_symbol, dtype='i')
        init = xp.full((memory.shape[0] * k, 1), start_symbol, dtype='i')
        outs = xp.array([[]] * batchsize * k, 'i')
        total_score = None
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               xp.array(subsequent_mask(ys.shape[1])))
            logit = model.generator(out[:, -1][:, None, :])
            topk, topk_score = get_topk(F.log_softmax(logit).data, k=k)
            outs, total_score = update_beam_state(outs, total_score, topk, topk_score, constant.EOS_ID)
            if i == 0:
                memory = F.repeat(memory, k, axis=0)
                src_mask = xp.repeat(src_mask, k, axis=0)
            ys = xp.concatenate([init, outs], axis=1)
            if xp.max(outs == constant.EOS_ID, axis=1).sum() == outs.shape[0]:
                # if all candidates meet eos, end decoding
                break
    result = finish_beam(outs, total_score, batchsize, constant.EOS_ID, length_penalty)
    return result

class v5SerialIterator(chainer.dataset.Iterator):

    """Dataset iterator that serially reads the examples.
    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.
    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.
    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.
    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguements: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.
    """

    def __init__(self, dataset, batch_size,
                 repeat=True, shuffle=None, order_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        if self._shuffle is not None:
            if order_sampler is not None:
                raise ValueError('`shuffle` is not `None` and a custom '
                                 '`order_sampler` is set. Please set '
                                 '`shuffle` to `None` to use the custom '
                                 'order sampler.')
            else:
                if self._shuffle:
                    order_sampler = ShuffleOrderSampler()
        else:
            if order_sampler is None:
                order_sampler = ShuffleOrderSampler()
        self.order_sampler = order_sampler

        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self._epoch_size

        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    new_order = self.order_sampler(self._order, i)
                    if len(self._order) != len(new_order):
                        raise ValueError('The size of order does not match '
                                         'the size of the previous order.')
                    self._order = new_order
                if rest > 0:
                    if self._order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend([self.dataset[index]
                                      for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / self._epoch_size
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
        if self.order_sampler:
            self._order = self.order_sampler(
                numpy.arange(len(self.dataset)), 0)
        else:
            self._order = None

    @property
    def _epoch_size(self):
        if self._order is None:
            return len(self.dataset)
        else:
            return len(self._order)

    @property
    def repeat(self):
        return self._repeat


class SemiBucketIterator(v5SerialIterator):
    """
    If shuffle is True, then read len(dataset) // 5 instances, sort & shuffle them, and create mini-batch.
    If shuffle is False, just create mini-batch.

    batchsize refers to number of tokens in the batch.
    Both source tokens and target tokens do not exceed the batchsize.
    E.g., batchsize 3500 --> source contains 3478 tokens, target contains 3238 tokens...
    """

    def __init__(self, **kwargs):
        super(SemiBucketIterator, self).__init__(**kwargs)

    def __sort_order(self, order, dataset):
        return sorted(order, key=lambda x: len(dataset[x][0]), reverse=True)

    def _create_order(self):
        batch_size = self.batch_size
        step = len(self.dataset) // 5  # step size for prefetch
        if self._shuffle:
            order = np.random.permutation(len(self.dataset)).tolist()
            prefetches = [self.__sort_order(order[i:i + step], self.dataset) for i in range(0, len(self.dataset), step)]
            order = self.__construct_batches(prefetches, batch_size)
            np.random.shuffle(order)
        else:
            order = list(range(0, len(self.dataset)))
            prefetches = [order[i:i + step] for i in range(0, len(self.dataset), step)]
            order = self.__construct_batches(prefetches, batch_size)
        return order

    def __construct_batches(self, prefetches, batch_size):

        def yield_batch(indices, dataset, batch_size):
            batch = []
            max_src = 0
            max_trg = 0
            for idx in indices:
                single_instance = dataset[idx]

                if isinstance(single_instance, tuple):  # target file exists in the training
                    src, trg = single_instance
                    # Always keep track of maximum length in the minibatch
                    # --> obtains pseudo-count of number of tokens in the minibatch
                    max_src = max(max_src, len(src))
                    max_trg = max(max_trg, len(trg) + 2)
                    src_elem = max_src * (len(batch) + 1)
                    trg_elem = max_trg * (len(batch) + 1)
                    # if adding new example exceeds the minibatch size
                    # then yield batch and create new one
                    if (src_elem > batch_size or trg_elem > batch_size) and len(batch) > 0:
                        yield batch
                        batch = [idx]
                        max_src = len(src)
                        max_trg = len(trg) + 2
                    else:
                        batch.append(idx)
                else:  # target file is not necessary in the test time
                    src = single_instance
                    # Always keep track of maximum length in the minibatch
                    # --> obtains pseudo-count of number of tokens in the minibatch
                    max_src = max(max_src, len(src))
                    src_elem = max_src * (len(batch) + 1)
                    # if adding new example exceeds the minibatch size
                    # then yield batch and create new one
                    if src_elem > batch_size and len(batch) > 0:
                        yield batch
                        batch = [idx]
                        max_src = len(src)
                    else:
                        batch.append(idx)
            yield batch

        out = []
        for prefetch in prefetches:
            for batch in yield_batch(prefetch, self.dataset, batch_size):
                out.append(batch)
        return out

    @property
    def current_epoch_progress(self):
        return self.current_position * 100 / len(self._order)

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self._order)

    def reset(self):
        self._order = self._create_order()

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + 1
        N = len(self._order)

        batch = [self.dataset[index] for index in self._order[i]]
        if i_end >= N:
            if self._repeat:
                self._order = self._create_order()
                self.current_position = 0
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__


def batchfy(batch, device):
    batch = Batch(*list(zip(*batch)))
    if device >= 0:
        batch.to_gpu(device_id=device)
    return batch


def train(args, out_dir):
    dpr = DataProcessor()
    dpr.load_vocab_from_path(args.enc_vocab_file, args.dec_vocab_file, args.limit_vocab_num)
    train_data = dpr.load_data_from_path(args.enc_data_file, args.dec_data_file, 'train', max_sent_length=args.max_sent_length)
    if args.semi:
        # load semi supervised data
        train_data_semi = dpr.load_data_from_path(args.enc_data_file_semi, args.dec_data_file_semi, 'train', max_sent_length=args.max_sent_length)
        train_data = list(train_data)
        if args.semi_balance_batch:
            random_idxs = np.random.permutation(len(train_data_semi))[:len(train_data)]
            train_data_semi = [train_data_semi[_idx] for _idx in random_idxs]
        for _ in train_data_semi:
            train_data.append(_)
        train_data = tuple(train_data)
    valid_data = dpr.load_data_from_path(args.enc_devel_data_file, args.dec_devel_data_file, 'dev')

    n_source_vocab = len(dpr.src_ivocab)
    n_target_vocab = len(dpr.trg_ivocab)
    print('n_source_vocab:', n_source_vocab)
    print('n_target_vocab:', n_target_vocab)
    print('train_data:', train_data[0])
    model = make_model(n_source_vocab, n_target_vocab, N=args.n_layer, d_model=args.d_model,
                       d_ff=args.d_ff, h=args.n_head, dropout=args.dropout,
                       share_embed=args.share_embed, smoothing=args.smoothing, use_vat=args.use_vat, eps=args.eps, perturbation_target=args.perturbation_target)
    optimizer = VaswaniAdam(factor=args.factor, model_size=args.d_model, warmup=args.warmup, beta1=args.beta1, beta2=args.beta2,
                            eps=args.adam_eps, inverse_square=args.inverse_square)
    optimizer.setup(model)

    # use_multi_gpu = len(args.gpus) > 1
    use_multi_gpu = True
    if use_multi_gpu:
        logger.info('Using Multiple GPUs ({})'.format(','.join([str(x) for x in args.gpus])))

        chunk_size = len(train_data) // len(args.gpus)
        train_data_splits = [train_data[i:i + chunk_size] for i in
                             list(range(0, len(train_data), chunk_size))[:len(args.gpus)]]

        for i, data_chunk in enumerate(train_data_splits):
            logger.info('Dataset #{} contains {} instances'.format(i, len(data_chunk)))
        logger.info('{} out of {} is used for training'.format(sum(len(x) for x in train_data_splits), len(train_data)))

        assert len(train_data_splits) == len(args.gpus)
        train_iters = [SemiBucketIterator(dataset=x, batch_size=args.batch_size, shuffle=True) for x in
                       train_data_splits]

        updater_fnc = MultiProcessParallelUpdaterMod
        if args.use_vat:
            updater_fnc = MultiProcessParallelUpdaterMod_VAD

        updater = updater_fnc(train_iters, optimizer=optimizer, devices=args.gpus,
                              converter=batchfy)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    short_term = (100, 'iteration')
    long_term = (1, 'epoch')
    valid_iter = SemiBucketIterator(dataset=valid_data, batch_size=args.batch_size, repeat=False,
                                    shuffle=False)
    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpus[0], converter=batchfy),
                   trigger=long_term)
    trainer.extend(extensions.snapshot_object(model, args.out+'_epoch_{.updater.epoch}.npz'), trigger=long_term)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.LogReport(trigger=short_term, log_name='chainer_report_iteration.log'),
                   trigger=short_term, name='iteration')
    trainer.extend(extensions.LogReport(trigger=long_term, log_name='chainer_report_epoch.log'),
                   trigger=long_term, name='epoch')
    trainer.extend(extensions.observe_value('lr', lambda x: x.updater.get_optimizer('main').lr),
                   trigger=short_term)
    entries = ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/perp', 'validation/main/perp']
    trainer.extend(extensions.PrintReport(entries=entries, log_report='iteration'), trigger=short_term)
    trainer.extend(extensions.PrintReport(entries=entries, log_report='epoch'), trigger=long_term)

    if args.resume:
        import glob
        models = sorted(glob.glob('/home/user/snapshots/' +'*.npz'))
        models = list(models)
        last_model = ''
        if len(models):
            last_model = list(models)[-1]
            chainer.serializers.load_npz(args.model, last_model)

    logger.info('Training Starts...')
    trainer.run()
    logger.info('Training Complete!')
    model.to_cpu()
    del model
    del trainer
    del train_data
    del optimizer
    del updater
    del train_iters
    del train_data_splits



def test(args, resource):
    dpr = DataProcessor()
    config = resource.config
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False
    src_vocab_path = config['enc_vocab_file']
    trg_vocab_path = config['dec_vocab_file']
    limit_vocab_num = config['limit_vocab_num']
    dpr.load_vocab_from_path(src_vocab_path, trg_vocab_path, limit_vocab_num)
    test_data = dpr.load_data_from_path(args.enc_data_file, data_type='test')
    test_iter = SemiBucketIterator(dataset=test_data, batch_size=args.batch_size, repeat=False, shuffle=False)
    n_source_vocab = len(dpr.src_ivocab)
    n_target_vocab = len(dpr.trg_ivocab)
    model = make_model(n_source_vocab, n_target_vocab, N=config['n_layer'], d_model=config['d_model'],
                       d_ff=config['d_ff'], h=config['n_head'], dropout=0.0, share_embed=config['share_embed'])
    if args.gpus[0] >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpus[0]).use()
        model.to_gpu(args.gpus[0])

    # logger.info('Load Model Params from: [{}]'.format(args.model))
    chainer.serializers.load_npz(args.model, model)

    # logger.info('Test Begins...')
    for i, batch in enumerate(test_iter):
        batch = Batch(batch, pad=constant.PAD_ID)
        if args.gpus[0] >= 0:
            batch.to_gpu(args.gpus[0])
        results = beam_decode(model, batch.src, batch.src_mask,
                              max_len=args.max_length, start_symbol=constant.BOS_ID, k=args.beam,
                              length_penalty=args.length_penalty)
        for result in results:
            print(u' '.join(dpr.trg_ivocab[i] for i in result['outs']), flush=True)
    # logger.info('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Chainer Implementation of Transformer Model (Port from PyTorch Version)')
    parser.add_argument('--mode', required=True, choices=['train', 'test'], type=str,
                        help='Working Mode: Train / Test')
    parser.add_argument('--batch-size', dest='batch_size', default=3500, type=int,
                        help='Maximum Number of Tokens Contained in Mini-batch sent to each GPU')
    parser.add_argument('--gpus', type=int, default=-1, nargs='*', help='GPU IDs (Negative Value Indicates CPU)')

    parser.add_argument('--epoch', default=20, type=int, help='Number of Epochs')
    parser.add_argument('--out-dir', dest='out_dir', default='./results', type=os.path.abspath,
                        help='Output Dir. Strongly recommend using "/work" as saving the model consumes huge storage.')
    parser.add_argument('--out', dest='out', default='model', type=str)
    parser.add_argument('--seed', default=12345, type=int, help='Random Seed')
    parser.add_argument('--layer', dest='n_layer', default=6, type=int,
                        help='Number of Stacking Encoder/Decoder Layers')
    parser.add_argument('--d-model', dest='d_model', default=512, type=int, help='Model Dimension')
    parser.add_argument('--d-ff', dest='d_ff', default=2048, type=int, help='Feed-Forward Dimension')
    parser.add_argument('--head', dest='n_head', default=8, type=int, help='Number of Heads for Multi-Headed Attention')
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help='Value for Label-Smoothing (0.0 indicates Normal Cross-Entropy)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout Rate')
    parser.add_argument('--share-embed', dest='share_embed', default=0, type=int, choices=[0, 1],
                        help='Share Source/Target Embedding Matrix or not')
    # Arguments for dataset & vocab
    parser.add_argument('--dataset', dest='dataset', type=str, default='',
                        choices=['iwslt2016-de-en', 'iwslt2016-en-de', 'iwslt2016-fr-en', 'iwslt2016-en-fr', 'wmt16-de-en', 'wmt16-en-de', 'wmt16-en-de-fairseq'])
    parser.add_argument('--enc-data-file', dest='enc_data_file', type=os.path.abspath,
                        help='filename of encoder (input)-side data for training')
    parser.add_argument('--dec-data-file', dest='dec_data_file', type=os.path.abspath,
                        help='filename of decoder (output)-side data for training')
    parser.add_argument('--enc-devel-data-file', dest='enc_devel_data_file', type=os.path.abspath,
                        help='filename of encoder (input)-side data for development data')
    parser.add_argument('--dec-devel-data-file', dest='dec_devel_data_file', type=os.path.abspath,
                        help='filename of decoder (output)-side data for development data')
    parser.add_argument('--enc-vocab-file', dest='enc_vocab_file', type=os.path.abspath,
                        help='filename of encoder (input)-side vocabulary')
    parser.add_argument('--dec-vocab-file', dest='dec_vocab_file', type=os.path.abspath,
                        help='filename of decoder (output)-side vocabulary')

    # Argumengts for Adam Optimizer & Vaswani's Magical Learning Rate Alternation Algorithm
    parser.add_argument('--warmup', dest='warmup', default=4000, type=int, help='Number of Steps for Warm-up')
    parser.add_argument('--beta1', dest='beta1', default=0.9, type=float, help='Beta1 Value of the Adam Optimizer')
    parser.add_argument('--beta2', dest='beta2', default=0.98, type=float, help='Beta2 Value of the Adam Optimizer')

    # Arguments for decoding
    parser.add_argument('--model', type=os.path.abspath, help='Path to the Pre-trained Model')
    parser.add_argument('--beam', type=int, default=10, help='Beam Width')
    parser.add_argument('--length-penalty', '-lp', dest='length_penalty', type=float, default=0.6,
                        help='Length Penalty α (0.0 < α < 1.0). Set 0.0 to disable. Refer https://arxiv.org/abs/1609.08144 for details')
    parser.add_argument('--max-length', '-ml', dest='max_length', type=int, default=60,
                        help='Maximum Lengths Allowed for Decoded Sequence')
    parser.add_argument('--max-sent-length', dest='max_sent_length', type=int, default=-1, help='Maximum Lengths Allowed for Decoded Sequence')
    parser.add_argument('--semi', dest='semi', default=0, type=int)
    parser.add_argument('--datatype', dest='datatype', default='', type=str)
    parser.add_argument('--use-vat', dest='use_vat', default=0, type=int)
    parser.add_argument('--eps', dest='eps', default=1.0, type=float)
    parser.add_argument('--perturbation-target', dest='perturbation_target', default=0, nargs='*', type=int, help='0:Enc, 1:Dec')
    parser.add_argument('--semi-balance-batch', dest='semi_balance_batch', default=0, type=int, help='0:all semi data, 1:balance batch data')

    parser.add_argument('--adam-eps', dest='adam_eps', default=1e-9, type=float, help='maybe 1e-8 in Scaling NMT')
    parser.add_argument('--factor', dest='factor', default=1.0, type=float, help='factor')
    parser.add_argument('--inverse-square', dest='inverse_square', default=0, type=int, help='inverse_square')
    parser.add_argument('--limit-vocab-num', dest='limit_vocab_num', default=-1, type=int, help='wmt limit vocab')
    parser.add_argument('--resume', dest='resume', default=0, type=int, help='resume')

    args = parser.parse_args()
    if isinstance(args.perturbation_target, int):
        args.perturbation_target = [args.perturbation_target]
    set_random_seed(args.seed, args.gpus)

    if args.dataset != '':
        from exp_setting import get_exp_dataset, set_exp_dataset
        enc_data_file_bk = args.enc_data_file
        def update_dateset_args():
            exp_dict = get_exp_dataset(name=args.dataset)
            # set_exp_dataset(exp_dict, args)
            args.enc_vocab_file = exp_dict['enc_vocab']
            args.dec_vocab_file = exp_dict['dec_vocab']
            args.enc_data_file = exp_dict['enc']
            args.dec_data_file = exp_dict['dec']
            args.enc_devel_data_file = exp_dict['enc_dev']
            args.dec_devel_data_file = exp_dict['dec_dev']
            if args.share_embed:
                args.enc_vocab_file = exp_dict['joint_vocab']
                args.dec_vocab_file = exp_dict['joint_vocab']
            if args.semi:
                # TODO: support Semi-supervised
                args.enc_data_file_semi = exp_dict['semi_enc_predict']
                args.dec_data_file_semi = exp_dict['semi_dec']
            if args.mode == 'test':
                # test mode
                args.dec_data_file = None
                if args.datatype == 'dev':
                    args.enc_data_file = exp_dict['enc_dev']
                elif args.datatype == 'eval1':
                    args.enc_data_file = exp_dict['enc_eval_1']
                elif args.datatype == 'eval2':
                    args.enc_data_file = exp_dict['enc_eval_2']
                elif args.datatype == 'semi_predict':
                    args.enc_data_file = enc_data_file_bk
        update_dateset_args()

    if args.mode == 'train':
        resource = Resource(args, train=True)
        resource.save_config_file()
        resource.save_vocab_file()
        resource.dump_command_info()
        resource.dump_git_info()
        resource.dump_python_info()
        train(args, resource.output_dir)
        resource.dump_library_info()  # This must be after training, as CUDA API will be called

    elif args.mode == 'test':
        resource = Resource(args, train=False)
        resource.load_config()
        assert len(args.gpus) == 1, 'Multi-GPU is not supported for decoding'
        test(args, resource)
    else:
        raise NotImplementedError
