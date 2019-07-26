# -*- coding: utf-8 -*-
import numpy as np
from logzero import logger

import constant

DATA_TYPE = ('train', 'dev', 'test')


class DataProcessor(object):
    def __init__(self):
        # dictionary for vocabulary
        # each variable is set by calling load_vocab_from_path
        self.src_vocab = None
        self.src_ivocab = None
        self.trg_vocab = None
        self.trg_ivocab = None
        self.logger = logger

    def load_data_from_path(self, src_data_path, trg_data_path=None, data_type='train', max_sent_length=-1):
        assert data_type in DATA_TYPE

        # Load Source Data
        assert self.src_vocab is not None
        self.logger.info('Loading {} - Source Data from [{}]'.format(data_type, src_data_path))
        src_data, src_ignore_ids = self._make_dataset(src_data_path, self.src_vocab, max_sent_length=max_sent_length)

        # (if given) Load Target Data
        if trg_data_path:
            assert self.trg_vocab is not None
            self.logger.info('Loading {} - Target Data from [{}]'.format(data_type, trg_data_path))
            trg_data, trg_ignore_ids = self._make_dataset(trg_data_path, self.trg_vocab, max_sent_length=max_sent_length)
            assert len(src_data) == len(trg_data)
            dataset = [(s, t) for s, t in zip(src_data, trg_data)]
        else:
            dataset = src_data
            trg_ignore_ids = []

        if len(src_ignore_ids) or len(trg_ignore_ids):
            ignore_ids = set(src_ignore_ids + trg_ignore_ids)
            dataset_filtered = []
            dataset = [d for i, d in enumerate(dataset) if i not in ignore_ids]

        return tuple(dataset)

    def load_vocab_from_path(self, src_vocab_path, trg_vocab_path, limit_vocab_num=-1):
        """
        Vocabulary file format: TOKEN_\t_INDEX
        """
        self.logger.info('Loading source vocabulary from {}'.format(src_vocab_path))
        self.logger.info('Loading target vocabulary from {}'.format(trg_vocab_path))

        self.src_vocab = {}
        self.src_vocab[constant.PAD_WORD] = len(self.src_vocab)
        self.src_vocab[constant.BOS_WORD] = len(self.src_vocab)
        self.src_vocab[constant.EOS_WORD] = len(self.src_vocab)
        self.src_vocab[constant.UNK_WORD] = len(self.src_vocab)

        def _load_vocab(vocab_path):
            words = []
            for x in open(vocab_path, encoding='utf-8'):
                if '\t' in x.strip():
                    w = x.strip().split('\t')[0]
                else:
                    w = x.strip()
                words.append(w)
            return words

        tmp_vocab = _load_vocab(src_vocab_path)
        if limit_vocab_num > 0:
            tmp_vocab = tmp_vocab[:limit_vocab_num]

        for w in tmp_vocab:
            self.src_vocab[w] = len(self.src_vocab)
        # self.src_vocab = {x.strip().split('\t')[0]: int(x.strip().split()[1]) for x in open(src_vocab_path, 'r')}
        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}

        self.trg_vocab = {}
        self.trg_vocab[constant.PAD_WORD] = len(self.trg_vocab)
        self.trg_vocab[constant.BOS_WORD] = len(self.trg_vocab)
        self.trg_vocab[constant.EOS_WORD] = len(self.trg_vocab)
        self.trg_vocab[constant.UNK_WORD] = len(self.trg_vocab)
        tmp_vocab = _load_vocab(trg_vocab_path)
        if limit_vocab_num > 0:
            tmp_vocab = tmp_vocab[:limit_vocab_num]
        for w in tmp_vocab:
            self.trg_vocab[w] = len(self.trg_vocab)
        # self.trg_vocab = {x.strip().split('\t')[0]: int(x.strip().split()[1]) for x in open(trg_vocab_path, 'r')}
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

    def _make_dataset(self, file_path, vocab, max_sent_length=-1):
        dataset = []
        ignore_ids = []
        with open(file_path, encoding='utf-8') as input_data:
            for i, line in enumerate(input_data):
                tokens = line.strip().split()
                if max_sent_length != -1 and len(tokens) > max_sent_length:
                    ignore_ids.append(i)

                if len(tokens) >= 1000:
                    self.logger.warn('Sentence containing more than 1000 tokens found')
                    self.logger.warn('You might want to check pre-processing script')
                xs = []
                xs += [vocab[constant.BOS_WORD]]
                xs += [vocab[t] if t in vocab else vocab[constant.UNK_WORD] for t in tokens]
                xs += [vocab[constant.EOS_WORD]]
                dataset.append(np.array(xs, dtype='i'))
        return dataset, ignore_ids

    def dump_vocab_to(self, dest, kind='source'):
        assert kind == 'source' or kind == 'target'
        if kind == 'source':
            vocab2idx = self.src_vocab
        else:
            vocab2idx = self.trg_vocab

        with open(dest, 'w') as fo:
            for vocab, idx in vocab2idx.items():
                fo.write('{}\t{}\n'.format(vocab, idx))
            self.logger.info('{} vocabulary is saved at {}'.format(kind, dest))
