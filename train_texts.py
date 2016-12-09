from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from DNC import DNC, DNC_output_len, DNC_input_len

import pickle
import math
import random
import chainer.iterators


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        y = self.l4(F.dropout(h3, train=self.train))
        return y


class DeepLSTM(chainer.Chain):

    def __init__(self, X, Y, n_hidden, train=True):
        super(DeepLSTM, self).__init__(
            l1=L.LSTM(X, n_hidden),
            l2=L.LSTM(n_hidden, n_hidden),
            # l3=L.LSTM(X, X),
            l4=L.Linear(n_hidden, Y),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        # self.l3.reset_state()

    def __call__(self, x):
        h1 = self.l1(F.dropout(x, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        # h3 = self.l3(F.dropout(h2, train=self.train))
        y = self.l4(F.dropout(h2, train=self.train))
        return y

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.


class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        self.current_index = 0
        self.current_progress = 0
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.dataset)

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        if not self.repeat and self.epoch > 0:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        cur_words = self.get_words()
        self.current_progress += 1
        next_words = self.get_words(ignore_label=-1)

        self.episode_end = False
        self.is_new_epoch = False

        if all([x == -1 for x in next_words]):
            self.episode_end = True
            self.current_progress = 0
            if self.current_index + self.batch_size >= len(self.dataset):
                self.current_index = 0
                self.is_new_epoch = True
                self.epoch += 1
                if self.shuffle:
                    random.shuffle(self.dataset)
            else:
                self.current_index += self.batch_size

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        l = max(len(x) for x in self.dataset[
                self.current_index:self.current_index + self.batch_size])
        n = math.ceil(len(self.dataset) / self.batch_size)

        # return self.epoch + (1 + self.current_index//self.batch_size +
        # (self.current_progress/l)) / n
        return self.epoch + (self.current_index / self.batch_size +
                             (self.current_progress / l)) / n

    def get_words(self, ignore_label=0):
        # It returns a list of current words.
        batch_range = range(self.current_index, min(
            self.current_index + self.batch_size, len(self.dataset)))
        p = self.current_progress
        return [self.dataset[i][p] if len(self.dataset[i]) > p else np.int32(ignore_label)
                for i in batch_range]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        # self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        self.current_index = serializer('current_index', self.current_index)
        self.current_progress = serializer(
            'current_progress', self.current_progress)


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len
        self.bprop_counter = 0
        self.loss = 0

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        # Concatenate the word IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        x, t = self.converter(batch, self.device)
        self.bprop_counter += 1

        # Compute the loss at this time step and accumulate it
        self.loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        if train_iter.episode_end:
            optimizer.target.predictor.reset_state()

        if train_iter.is_new_epoch or self.bprop_counter >= self.bprop_len:
            optimizer.target.cleargrads()  # Clear the parameter gradients
            self.loss.backward()  # Backprop
            self.loss.unchain_backward()  # Truncate the graph
            optimizer.update()  # Update the parameters

            self.loss = 0
            self.bprop_counter = 0


def compute_perplexity(result):
    # Routine to rewrite the result dictionary of LogReport to add perplexity
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=40,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('data')
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        train, i_to_c, c_to_i = pickle.load(f)
        train = [np.array(x, dtype=np.int32) for x in train]

    n_vocab = len(i_to_c)
    print('#vocab =', n_vocab)

    val = None
    if args.test:
        train = train[:35]
        val = train[:35]

    train_iter = ParallelSequentialIterator(train, args.batchsize)

    # Prepare a model
    R = 16
    W = 64
    rnn = DNC(n_vocab, n_vocab, 64, W, R, DeepLSTM(DNC_input_len(n_vocab, W, R),
                                                   DNC_output_len(n_vocab, W, R), 500))  # RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.RMSprop()  # SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    if val is not None:
        val_iter = ParallelSequentialIterator(
            val, args.batchsize, repeat=False)
        eval_model = model.copy()  # Model with shared params and distinct states
        eval_rnn = eval_model.predictor
        eval_rnn.train = False
        trainer.extend(extensions.Evaluator(
            val_iter, eval_model, device=args.gpu,
            # Reset the RNN state at the beginning of each evaluation
            eval_hook=lambda _: eval_rnn.reset_state()))

    trainer.extend(extensions.LogReport(postprocess=compute_perplexity))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ))
    trainer.extend(extensions.ProgressBar(update_interval=5))

    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
