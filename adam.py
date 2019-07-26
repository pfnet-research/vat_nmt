# -*- coding: utf-8 -*-
from __future__ import division

import math

import chainer
from chainer import optimizer


def _learning_rate(hp, t):
    """
    Computes learning rate (alpha), According to Vaswani's Magical Warm-up Technique
    """
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    lr = hp.factor * \
         (hp.model_size ** (-0.5) *
          min(t ** (-0.5), t * hp.warmup ** (-1.5)))
    return lr * math.sqrt(fix2) / fix1



def _learning_rate_fairseq(hp, t):
    """
    Computes learning rate (alpha), According to Vaswani's Magical Warm-up Technique
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    """
    warmup_init_lr = 1e-07
    # lr = 0.0005
    lr = 0.001
    min_lr =  1e-09
    warmup_end_lr = lr
    warmup_updates = hp.warmup
    decay_factor = warmup_end_lr * warmup_updates**0.5
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')

    num_updates = t
    if num_updates < warmup_updates:
        lr = warmup_init_lr + num_updates*lr_step
    else:
        lr = decay_factor * num_updates**-0.5
    return lr


class VaswaniAdamRule(chainer.optimizers.adam.AdamRule):
    def __init__(self, hp, inverse_square=False):
        super(VaswaniAdamRule, self).__init__(hp)
        self.inverse_square = inverse_square

    @property
    def lr(self):
        if self.inverse_square:
            return _learning_rate_fairseq(self.hyperparam, self.t)
        else:
            return _learning_rate(self.hyperparam, self.t)


class VaswaniAdam(chainer.optimizers.Adam):
    def __init__(self, factor, warmup, model_size, inverse_square=False, **kwargs):
        super(VaswaniAdam, self).__init__(**kwargs)
        # Vaswani
        self.hyperparam.factor = factor
        self.hyperparam.warmup = warmup
        self.hyperparam.model_size = model_size
        self.inverse_square = inverse_square

    def create_update_rule(self):
        return VaswaniAdamRule(self.hyperparam, inverse_square=self.inverse_square)

    # Vaswani
    factor = optimizer.HyperparameterProxy('factor')
    warmup = optimizer.HyperparameterProxy('warmup')
    model_size = optimizer.HyperparameterProxy('model_size')

    @property
    def lr(self):
        if self.inverse_square:
            return _learning_rate_fairseq(self.hyperparam, self.t)
        else:
            return _learning_rate(self.hyperparam, self.t)
