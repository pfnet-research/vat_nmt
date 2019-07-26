# -*- coding: utf-8 -*-
import argparse
import os
from collections import defaultdict, Counter

from logzero import logger

import constant


def build_vocabulary(file_path, limit, prefix, suffix, out_dir):
    vocab_count = Counter()
    with open(file_path, 'r') as fi:
        for line in fi:
            tokens = line.rstrip().split()
            for token in tokens:
                vocab_count[token] += 1

    word2id = defaultdict(lambda: len(word2id))
    word2id[constant.PAD_WORD]
    word2id[constant.BOS_WORD]
    word2id[constant.EOS_WORD]
    word2id[constant.UNK_WORD]

    # ここにspecial token
    assert word2id[constant.PAD_WORD] == constant.PAD_ID
    assert word2id[constant.BOS_WORD] == constant.BOS_ID
    assert word2id[constant.EOS_WORD] == constant.EOS_ID
    assert word2id[constant.UNK_WORD] == constant.UNK_ID
    for token, count in vocab_count.most_common(limit):
        word2id[token]

    file_name = '{}.{}.dict'.format(prefix, suffix)
    destination = os.path.join(out_dir, file_name)
    logger.info('Creating vocab file at [{}]'.format(destination))
    with open(destination, 'w') as fo:
        for token, index in word2id.items():
            fo.write('{}\t{}\n'.format(token, index))


def build_single_vocabulary(src_file, trg_file, prefix, out_dir):
    word2id = defaultdict(lambda: len(word2id))
    word2id[constant.PAD_WORD]
    word2id[constant.BOS_WORD]
    word2id[constant.EOS_WORD]
    word2id[constant.UNK_WORD]
    # word2id[constants.FILLER_WORD]
    assert word2id[constant.PAD_WORD] == constant.PAD_ID
    assert word2id[constant.UNK_WORD] == constant.UNK_ID
    assert word2id[constant.BOS_WORD] == constant.BOS_ID
    assert word2id[constant.EOS_WORD] == constant.EOS_ID

    with open(src_file, 'r') as fi:
        for line in fi:
            tokens = line.rstrip().split()
            for token in tokens:
                word2id[token]

    with open(trg_file, 'r') as fi:
        for line in fi:
            tokens = line.rstrip().split()
            for token in tokens:
                word2id[token]

    file_name = '{}.src.dict'.format(prefix)
    destination = os.path.join(out_dir, file_name)
    logger.info('Creating src vocab file at [{}]'.format(destination))
    with open(destination, 'w') as fo:
        for token, index in word2id.items():
            fo.write('{}\t{}\n'.format(token, index))

    file_name = '{}.trg.dict'.format(prefix)
    destination = os.path.join(out_dir, file_name)
    logger.info('Creating trg vocab file at [{}]'.format(destination))
    with open(destination, 'w') as fo:
        for token, index in word2id.items():
            fo.write('{}\t{}\n'.format(token, index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocabulary file builder")
    parser.add_argument('-v', '--vocab-limit', dest='limit', default=50000, type=int,
                        help='max source & target vocabulary size')
    parser.add_argument('--src-limit', dest='src_limit', default=0, type=int, help='max source vocabulary size')
    parser.add_argument('--trg-limit', dest='trg_limit', default=0, type=int, help='max target vocabulary size')
    parser.add_argument('--src-file', dest='src', required=True, type=str, help='path to source file')
    parser.add_argument('--trg-file', dest='trg', required=True, type=str, help='path to target file')
    parser.add_argument('--prefix', dest='prefix', default='vocab', type=str, help='prefix of the vocabulary file')
    parser.add_argument('--out', '-o', dest='out', default='', type=os.path.abspath, help='output dir')
    parser.add_argument('--single', choices=[0, 1], default=0, type=int,
                        help='Create single unified vocabulary set for sharing embedding')
    args = parser.parse_args()

    # overwrite args.limit if limit is given specifically
    src_limit = args.src_limit if args.src_limit != 0 else args.limit
    trg_limit = args.trg_limit if args.trg_limit != 0 else args.limit

    logger.info('Source File: [{}]'.format(args.src))
    logger.info('Target File: [{}]'.format(args.trg))
    if args.single:
        logger.info('Generating unified vocabulary set (this is typically for sharing embedding layer).')
        build_single_vocabulary(src_file=args.src, trg_file=args.trg, prefix=args.prefix, out_dir=args.out)
    else:
        logger.info('Generating separate vocabulary set.')
        build_vocabulary(limit=src_limit, file_path=args.src, prefix=args.prefix, suffix='src', out_dir=args.out)
        build_vocabulary(limit=trg_limit, file_path=args.trg, prefix=args.prefix, suffix='trg', out_dir=args.out)
    logger.info('Done.')
