from __future__ import print_function

import os
import time
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils


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

    print("X")
    print(X[1])
    print("Y")
    print(y[0])
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

    # Make binary problem
    #print([x for x in X_test])
    #y_train = [[1,0] if x >= 0 else [0,1] for x in y_train]
    print(y_test.shape)
    #y_test = [[1,0] if x >= 0 else [0,1] for x in y_test]

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    #y_train = y_train.reshape((-1, 1))
    #y_test = y_test.reshape((-1, 1))
    q = np.atleast_3d([timeseries[-window_size:]])

    return X_train, y_train, X_test, y_test, q


def preprocess(path):
    with open(path) as f:
      lines = f.readlines()

    if os.path.exists('vecs.npy'):
      npvecs = np.load('vecs.npy')
    else:
      words = ' '.join(lines[:1000]).replace('\n', ' ') # just first 1000
      words = nltk.word_tokenize(words)
      words = [x for x in words if len(x) > 0]
      print(words)
      print(len(words))
      
      f = load_model('wiki.en.bin')

      vecs = []
      for w in words:
          vec = f.get_word_vector(w)
          vecs.append(vec) 
          print((w, vec))

      npvecs = np.asarray(vecs)
      np.save('vecs', npvecs)

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
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape)
    print(y_train.shape)

    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

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

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    print("Training")

    list_train_loss = []
    list_test_loss = []
    list_learning_rate = []

    for e in range(nb_epoch):

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = X_train.shape[0] / split_size
        arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

        l_train_loss = []
        start = time.time()

        for batch_idx in arr_splits:

            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])

        test_logloss, test_acc = model.evaluate(X_test,
                                                Y_test,
                                                verbose=0,
                                                batch_size=64)
        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_test_loss.append([test_logloss, test_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log/experiment_log_cifar10.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NLP experiment')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
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

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

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
