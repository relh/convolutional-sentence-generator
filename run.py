from __future__ import print_function

from collections import defaultdict
import os
import pickle
import time
import json
import random
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
checkpointer = ModelCheckpoint(filepath=name+'.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000001, verbose=1)


def munge_data(timeseries, words, indices, window_size):
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    #print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    #print(timeseries.shape)

    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_2d(np.array([(start, start + window_size) for start in range(0, timeseries.shape[0] - window_size)]))

    # y is now the index in the 10,000 output softmax
    y = np.asarray([indices[x] for x in words[window_size:]])
    #y = timeseries[window_size:]

    #print('\n\nInput features:', X, '\n\nOutput labels:', y, sep='\n')
    test_size = int(0.05 * nb_samples) # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

    X_train = X_train #np.expand_dims(X_train, axis=3)
    X_test = X_test #np.expand_dims(X_test, axis=3)

    return X_train, y_train, X_test, y_test

def load_data(path):
    with open(path) as f:
      words = tokenize(f.read())

    counts = defaultdict(int)
    for word in words:
      counts[word] += 1 

    if os.path.exists('indices.p'):
      indices = pickle.load( open( "indices.p", "rb" ) )
    else:
      indices = defaultdict(int)
      i = 0
      for word in counts.keys():
        indices[word] = i
        i += 1
      print("i is: " + str(i))
      pickle.dump( indices, open( "indices.p", "wb" ) )

    words = [x for x in words if len(x) > 0]
    return words, indices

def preprocess(path, words):
    vec_path = path.split('.')[0]
    if os.path.exists(vec_path+'.npy'):
      np_vecs = np.load(vec_path+'.npy')
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

    window_size = 100
    nb_classes = 10000 #len(np.unique(y_train))
    img_dim = (window_size, 300) #X_train.shape[1:]

    words, indices = load_data(path)
    timeseries = preprocess(path, words)
    #X_train, y_train, X_test, y_test = munge_data(timeseries, words, indices, window_size)

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

      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['mae', 'accuracy'])

    #if plot_architecture:
    #    from keras.utils.visualize_util import plot
    #    plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    train(model, timeseries, indices, words, window_size, args)
    #test(model, X_train, y_train, X_test, y_test)


def train(model, timeseries, indices, words, window_size, args):
    # convert class vectors to binary class matrices
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    def generator(timeseries, batch_size, top, bot=1):
     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((batch_size, 100, 300))

     while True:
       batch_labels = np.zeros((batch_size,10000))
       for i in range(batch_size):
          # choose random index in features
          index = random.choice(range(bot, top)) #len(timeseries)-window_size,1)
          X = np.atleast_2d(np.array([timeseries[start:start + window_size] for start in range(index, index+1)]))
          batch_features[i] = X

          # y is now the index in the 10,000 output softmax
          batch_labels[i][indices[words[index+window_size]]] = 1.0

       yield batch_features, batch_labels

    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.

    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    top = len(timeseries)-window_size-int(len(timeseries)*0.05)
    model.fit_generator(generator(timeseries, args.batch_size, top), steps_per_epoch=10000, epochs=args.nb_epoch, validation_data=generator(timeseries, args.batch_size, len(timeseries)-window_size, top), validation_steps=500, callbacks=[lr_reducer, checkpointer], shuffle=True)
    #model.save(name+'.h5')

def test(model, X_train, y_train, X_test, y_test):
    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    with open('out.csv', 'w') as f:
      i = 0
      true = 0
      for actual, predicted in zip(y_test, pred):
        i += 1
        f.write(str(actual) + ',')
        f.write(str(np.argmax(predicted)) + '\n')
        if actual == np.argmax(predicted):
          true += 1
        if i % 100 == 0:
          print("True: {} / {}".format(true, i))

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
    for key, value in parser.parse_args()._get_kwargs():
        print(key, value)

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
