#!/usr/bin/env python
"""
Example of using Keras to implement a 1D convolutional neural network (CNN) for timeseries prediction.
"""

from __future__ import print_function, division

import argparse
import os
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras.layers import Conv1D, AtrousConv1D, Flatten, Dense, \
    Input, Lambda, merge, Activation, MaxPooling1D, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam
from keras import regularizers

import sys
sys.path.insert(0,'./DenseNet/')

import numpy as np
import pandas as pd

from DenseNet import * 

# Which network
name = 'v1' 

#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=name+'.h5', verbose=1, save_best_only=True)


def parse_arg(): # parses all the command line arguments
    parser = argparse.ArgumentParser('nlp')
    parser.add_argument('mode', type=str, default='train', help='mode')
    args = parser.parse_args()
    return args


def wavenetBlock(n_atrous_filters, atrous_filter_size, dilation_rate):
    def f(input_):
        residual = input_
        tanh_out = AtrousConv1D(n_atrous_filters, atrous_filter_size,
                                       dilation_rate=dilation_rate,
                                       padding='same',
                                       activation='tanh',
                                       kernel_regularizer=regularizers.l2(0.01),
                                       activity_regularizer=regularizers.l1(0.01))(input_)
        sigmoid_out = AtrousConv1D(n_atrous_filters, atrous_filter_size,
                                       dilation_rate=dilation_rate,
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_regularizer=regularizers.l2(0.01),
                                       activity_regularizer=regularizers.l1(0.01))(input_)
        tanh_out = Dropout(0.3)(tanh_out)
        sigmoid_out = Dropout(0.3)(sigmoid_out)
        merged = merge([tanh_out, sigmoid_out], mode='mul')
        skip_out = Conv1D(1, 1, activation='relu', padding='same')(merged)
        out = merge([skip_out, residual], mode='sum')
        return out, skip_out
    return f

def get_basic_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 10, 2)(input_) #5
    skip_connections = [B]
    for i in range(15):
        A, B = wavenetBlock(64, 4, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = merge(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Conv1D(1, 1)(net) # activation='relu'
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = Conv1D(1, 1)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = Flatten()(net)
    net = Dense(2, activation='softmax')(net)
    model = Model(input=input_, output=net)
    opt = Nadam(lr=0.002)
    #model.compile(loss='mse', optimizer=opt, metrics=['mae', 'accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    return model


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

def load_network(window_size):
    if os.path.exists(name+'.h5'):
      model = load_model(name+'.h5')
    else:
      model = get_basic_generative_model(window_size)
    print('\n\nModel with input size {}'.format(model.input_shape))
    return model

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
    y_train = [[1,0] if x >= 0 else [0,1] for x in y_train]
    print(y_test.shape)
    y_test = [[1,0] if x >= 0 else [0,1] for x in y_test]

    #y_train = y_train.reshape((-1, 1))
    #y_test = y_test.reshape((-1, 1))
    q = np.atleast_3d([timeseries[-window_size:]])


    """
    X_test[X_test > 0] = [1, 0]
    X_test[X_test < 0] = [0, 1]
    y_test[y_test > 0] = [1, 0]
    y_test[y_test < 0] = [0, 1]
    """
    print(X_test)
    return X_train, y_train, X_test, y_test, q

def train(model, X_train, y_train, X_test, y_test, q):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.

    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_reducer, checkpointer], shuffle=True)
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


def main(args):
    """Prepare input data, build model, evaluate."""
    np.set_printoptions(threshold=25)
    window_size = 2048

    # Preprocess data
    timeseries = preprocess('big.txt')
    print(timeseries.shape)

    X_train, y_train, X_test, y_test, q = load_data(timeseries, window_size)

    print('\nSimple single timeseries vector prediction')
    # Load model
    model = load_network(window_size)

    print(args.mode)
    if True or args.mode is "train":
      train(model, X_train, y_train, X_test, y_test, q)
    else:
      test(model, X_train, y_train, X_test, y_test, q)

def confusion():
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    model.load_weights(name+'.h5')
    pred = model.predict(np.array(X_test))
    C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
    print(C / C.astype(np.float).sum(axis=1))

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def preprocess(path):
    data = pd.read_csv(path, sep='\t').get_values()[:, 1:]

    c = np.diff(data[:, 0]) / (np.abs(data[:-1, 0])) * 100
    h = np.diff(data[:, 1]) / (np.abs(data[:-1, 1])) * 100
    l = np.diff(data[:, 2]) / (np.abs(data[:-1, 2])) * 100
    o = np.diff(data[:, 3]) / (np.abs(data[:-1, 3])) * 100
    qv = data[:, 4]
    v = data[:, 5]
    wa = np.diff(data[:, 6]) / (np.abs(data[:-1, 6])) * 100

    # Max
    for data in [c, h, l, o, qv, v, wa]:
      std_dev = np.std(data)
      print("Std Dev: \t" + str(std_dev))
      print("Max: \t\t" + str(max(data)))
      print("Min: \t\t" + str(min(data)))
      data[data > 3*std_dev] = 3*std_dev
      data[data < 3*-std_dev] = 3*-std_dev
      print(max(data))
      print(min(data))

    print(data)
    return data

def textify():
    from fastText import load_model

    f = load_model('wiki.en.bin')

    print(f.get_word_vector("London"))
    print(f.get_word_vector("London").shape)

if __name__ == '__main__':
    image_dim = (224, 224, 3)
    model = DenseNet(classes=10, input_shape=image_dim, depth=40, growth_rate=12, 
            bottleneck=True, reduction=0.5)
    args = parse_arg()
    main(args)
