#!/usr/bin/env python
"""Example to generate text from a recurrent neural network language model.

This code is ported from following implementation.
https://github.com/longjie/chainer-char-rnn/blob/master/sample.py

"""
import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from DNC import DNC, DNC_output_len, DNC_input_len

import train_texts
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='model data, saved by train_ptb.py')
    parser.add_argument('--primetext', '-p', type=str, required=True,
                        default='',
                        help='base text data, used for text generation')
    parser.add_argument('--seed', '-s', type=int, default=123,
                        help='random seeds for text generation')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='number of units')
    parser.add_argument('--sample', type=int, default=1,
                        help='negative value indicates NOT use random choice')
    parser.add_argument('--length', type=int, default=2000,
                        help='length of the generated text')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('data')
    args = parser.parse_args()

    np.random.seed(args.seed)

    xp = cuda.cupy if args.gpu >= 0 else np

    with open(args.data, 'rb') as f:
        _, i_to_c, c_to_i = pickle.load(f)

    vocab = c_to_i
    ivocab = i_to_c
    # should be same as n_units , described in train_ptb.py
    n_units = args.unit
    n_vocab = len(i_to_c)
    R = 16
    W = 64
    lm = DNC(n_vocab, n_vocab, 64, W, R, train_texts.DeepLSTM(DNC_input_len(n_vocab, W, R), DNC_output_len(
        n_vocab, W, R), 500))  # train_texts.RNNForLM(len(i_to_c), n_units, train=False)
    model = L.Classifier(lm)

    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    model.predictor.reset_state()

    primetext = args.primetext
    if isinstance(primetext, six.binary_type):
        primetext = primetext.decode('utf-8')

    if primetext in vocab:
        prev_word = chainer.Variable(xp.array([vocab[primetext]], xp.int32))
    else:
        print('ERROR: Unfortunately ' + primetext + ' is unknown.')
        exit()

    prob = F.softmax(model.predictor(prev_word))
    sys.stdout.write(primetext)

    for i in six.moves.range(args.length):
        prob = F.softmax(model.predictor(prev_word))
        if args.sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        if index == 0:
            break

        sys.stdout.write(ivocab[index])

        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
