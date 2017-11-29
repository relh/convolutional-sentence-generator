from __future__ import print_function

import os
import time
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

import nltk
from fastText import load_model

name = 'v1'
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=name+'.h5', verbose=1, save_best_only=True)


def make_timeseries_instances(timeseries, window_size):
    """Make input features and prediction targets from a `timeseries` for use in machine learning.

    :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
      ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
      corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
      to predict a hypothetical next (unprovided) value in the `timeseries`.
    :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
      row) and the series is axis 1 (the column).
    :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
    """
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]

    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q


def load_data(timeseries, window_size):
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)

    X, y, q = make_timeseries_instances(timeseries, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
    test_size = int(0.05 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    q = np.atleast_3d([timeseries[-window_size:]])

    return X_train, y_train, X_test, y_test, q


def preprocess(path):
    with open(path) as f:
      lines = f.readlines()

    if os.path.exists('big_vecs.npy'):
      npvecs = np.load('big_vecs.npy')
    else:
      words = ' '.join(lines).replace('\n', ' ') # lines[:1000] just first 1000
      words = nltk.word_tokenize(words)
      words = [x for x in words if len(x) > 0]
      words_len = len(words)
      
      f = load_model('wiki.en.bin')

      vecs = []
      for i, w in enumerate(words):
          vec = f.get_word_vector(w)
          vecs.append(vec) 
          if i % 1000 == 0:
            print("{} / {}".format(i, words_len))

      npvecs = np.asarray(vecs)
      np.save('big_vecs', npvecs)

    print(npvecs.shape)
    return npvecs


def run(batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                plot_architecture):
    """ Run Conv NLP experiments

    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture

    """

    ###################
    # Data processing #
    ###################

    window_size = 100 
    timeseries = preprocess('big.txt')
    print(timeseries.shape)

    X_train, y_train, X_test, y_test, q = load_data(timeseries, window_size)
    print(X_train.shape)
    print(y_train.shape)
    print('---')

    nb_classes = 300 #len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = y_train #np_utils.to_categorical(y_train, nb_classes)
    Y_test = y_test #np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    ###################
    # Construct model #
    ###################

    model = densenet.DenseNet(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #opt = Nadam(lr=0.002)

    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mae', 'accuracy'])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    train(model, X_train, y_train, X_test, y_test, q, args)


def train(model, X_train, y_train, X_test, y_test, q, args):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.

    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    model.fit(X_train, y_train, epochs=args.nb_epoch, batch_size=args.batch_size, validation_data=(X_test, y_test), callbacks=[lr_reducer, checkpointer], shuffle=True)
    model.save(name+'.h5')

    #test(X_train, y_train, X_test, y_test, q)

def test(model, X_train, y_train, X_test, y_test, q):
    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    with open('out.csv', 'w') as f:
      for actual, predicted in zip(y_test, pred.squeeze()):
        f.write(str(actual)+',') 
        f.write(str(predicted)+'\n')
        print(actual.squeeze(), predicted, sep='\t')
    print('next', model.predict(q).squeeze(), sep='\t')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NLP experiment')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    run(args.batch_size,
                args.nb_epoch,
                args.depth,
                args.nb_dense_block,
                args.nb_filter,
                args.growth_rate,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay,
                args.plot_architecture)
