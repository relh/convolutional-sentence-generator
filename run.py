from __future__ import print_function

from collections import defaultdict
import os
import time
import json
import argparse
import numpy as np
import keras.backend as K

from keras.models import load_model as keras_load_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

import nltk
from fastText import load_model
from fastText import tokenize
from gensim.models.wrappers import FastText

import densenet

name = 'v1'
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=name+'.h5', verbose=1, save_best_only=True)


def munge_data(timeseries, words, indices, window_size, epoch_size):
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    print(timeseries.shape)

    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    assert 0 < epoch_size < timeseries.shape[0]-window_size
    X = np.atleast_2d(np.array([timeseries[start:start + window_size] for start in range(0, epoch_size)])) #timeseries.shape[0] - window_size)]))

    # y is now the index in the 10,000 output softmax
    y = np.asarray([indices[x] for x in words[window_size:epoch_size+window_size]])
    #y = timeseries[window_size:epoch_size+window_size]
    print(y)

    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', 0, sep='\n')
    test_size = int(0.05 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

    X_train = X_train #np.expand_dims(X_train, axis=3)
    X_test = X_test #np.expand_dims(X_test, axis=3)
    q = np.atleast_3d([timeseries[-window_size:]])

    return X_train, y_train, X_test, y_test

def load_data(path):
    indices = defaultdict(int)
    counts = defaultdict(int)
    with open(path) as f:
      words = tokenize(f.read())

    for word in words:
      counts[word] += 1 

    i = 0
    for word in counts.keys():
      indices[word] = i
      i += 1
    print("i is: " + str(i))

    words = [x for x in words if len(x) > 0]
    return words, indices

def preprocess(path, words):
    vec_path = path.split('.')[0]
    if os.path.exists(vec_path+'.npy'):
      np_vecs = np.load(vec_path+'.npy')
      print(np_vecs)
    else:
      words_len = len(words)
      
      f = load_model('wiki.en.bin')

      vecs = []
      for i, w in enumerate(words):
          vec = f.get_word_vector(w)
          vecs.append(vec) 
          if i % 1000 == 0:
            print("{} / {}".format(i, words_len))

      np_vecs = np.asarray(vecs)
      np.save(vec_path, np_vecs)
    return np_vecs


def run(batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                plot_architecture,
                path):
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

    epoch_size = 1000
    window_size = 100 
    nb_classes = 10000 #len(np.unique(y_train))
    words, indices = load_data(path)
    timeseries = preprocess(path, words)
    print(timeseries.shape)

    X_train, y_train, X_test, y_test = munge_data(timeseries, words, indices, window_size, epoch_size)
    print(X_train.shape)
    print(y_train.shape)
    print('---')

    img_dim = X_train.shape[1:]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print(Y_train.shape)
    print(Y_test.shape)
    print('---')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    ###################
    # Construct model #
    ###################

    if os.path.exists(name+'.h5'):
      model = keras_load_model(name+'.h5')
    else:
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

    #train(model, X_train, y_train, X_test, y_test, q, args)

    test(model, X_train, y_train, X_test, y_test, q)


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
    #ft = FastText()
    #wv = ft.load_fasttext_format('wiki.en.bin')
    print('\n\nactual', 'predicted', sep='\t')
    with open('out.csv', 'w') as f:
      i = 0
      for actual, predicted in zip(y_test, pred.squeeze()):
        i += 1
        f.write(str(wv.similar_by_vector(actual)) + ',')
        f.write(str(wv.similar_by_vector(predicted)) + '\n')
        if i % 100 == 0:
          print("{} / {}".format(i, 1234))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NLP experiment')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=1, type=int,
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
    parser.add_argument('--path', type=str, default='data/train.txt',
                        help='Specify file path to train on')

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
                args.plot_architecture,
                args.path)
