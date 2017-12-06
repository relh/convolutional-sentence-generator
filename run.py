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
from keras.optimizers import Nadam
from keras.utils import np_utils

from fastText import load_model
from fastText import tokenize

import densenet

version = 'v23'
name = 'models/' + version
# val_perplexity
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=name+'.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=0.000001, verbose=1)


def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity


def cel_perplexity(model, timeseries, indices, words, args):
    accum = 0
    words_len = len(words)-args.window_size
    batches = words_len / args.batch_size
    print(batches)
    for start in range(0, batches):
      idx = start*args.batch_size
      inp = np.array([timeseries[i:i+args.window_size] for i in range(idx, idx+args.batch_size)])
      label = np_utils.to_categorical(np.asarray([indices[x] for x in words[idx+args.window_size:idx+args.window_size+args.batch_size]]), args.nb_classes)
      
      pred, _, _ = model.evaluate(inp, label)
      accum += pred 
      if start % 5 == 0:
        print("{} / {}. Perplexity so far: {}".format(start, batches, np.exp(-accum / (start*args.batch_size+1))))
    print(accum)
    avg = accum / (batches) 
    print(avg)
    perplex = np.exp(2.0, avg)
    print(perplex)


def nll_perplexity(model, timeseries, indices, words, args):
    accum = 0
    words_len = len(words)-args.window_size
    batches = words_len / args.batch_size
    print(batches)
    for start in range(0, batches):
      idx = start*args.batch_size
      inp = np.array([timeseries[i:i+args.window_size] for i in range(idx, idx+args.batch_size)])
      label = np.asarray([indices[x] for x in words[idx+args.window_size:idx+args.window_size+args.batch_size]])
      
      pred = model.predict(inp)
      lp = np.log(pred)
      for i, ent in enumerate(lp):
        accum += ent[label[i]]
      if start % 5 == 0:
        print("{} / {}. Perplexity so far: {}".format(start, batches, np.exp(-accum / (start*args.batch_size+1))))
    accum = -accum
    print(accum)
    avg = accum / words_len 
    print(avg)
    perplex = np.exp(avg)
    print(perplex)


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


def run(args):
    ###################
    # Construct model #
    ###################

    if os.path.exists(name+'.h5'):
      model = keras_load_model(name+'.h5', custom_objects={'perplexity': perplexity})
      model.summary()
    else:
      model = densenet.DenseNet(args.nb_classes,
                                args.img_dim,
                                args.depth,
                                args.nb_dense_block,
                                args.growth_rate,
                                args.nb_filter,
                                dropout_rate=args.dropout_rate,
                                weight_decay=args.weight_decay)
      # Model output
      model.summary()

      # Build optimizer
      opt = Nadam(lr=args.learning_rate)

      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=[perplexity, 'mae', 'accuracy'])

    ####################
    # Network training #
    ####################

    words, indices = load_data(args.train_path)
    timeseries = preprocess(args.train_path, words)
    train(model, timeseries, indices, words, args)

    words, _ = load_data(args.test_path)
    timeseries = preprocess(args.test_path, words)
    #cel_perplexity(model, timeseries, indices, words, args)
    #nll_perplexity(model, timeseries, indices, words, args)


def generator(timeseries, indices, words, args, top=-1, bot=1):
    print(top)
    if top < 0:
      top = len(timeseries)
    while True:
        # choose random index in features
        start = random.choice(range(bot, top)) #len(timeseries)-window_size,1)
        batch_features = np.array([timeseries[i:i+args.window_size] for i in range(start, start+args.batch_size)])
        batch_labels = np_utils.to_categorical(np.asarray([indices[x] for x in words[start+args.window_size:start+args.window_size+args.batch_size]]), args.nb_classes)

        yield batch_features, batch_labels


def train(model, timeseries, indices, words, args):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.

    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    top = len(timeseries)-args.window_size-args.batch_size-int(len(timeseries)*0.05)
    print("--- Model Version: {} ---".format(name))
    print("Number of words in training set: {}".format(top))
    model.fit_generator(generator(timeseries, indices, words, args, top, 1), steps_per_epoch=args.epoch_steps, epochs=args.nb_epoch, validation_data=generator(timeseries, indices, words, args, len(timeseries)-args.window_size-args.batch_size, top), validation_steps=args.val_steps, callbacks=[lr_reducer, checkpointer], shuffle=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NLP experiment')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=25000, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=13,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=512,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=64,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')
    parser.add_argument('--test_path', type=str, default='data/train.txt',
                        help='Specify file path to train on')
    parser.add_argument('--train_path', type=str, default='data/test.txt',
                        help='Specify file path to test on')
    parser.add_argument('--window_size', type=int, default=100,
                        help='How many words to use as context')
    parser.add_argument('--nb_classes', type=int, default=10000,
                        help='Number of classes')
    parser.add_argument('--img_dim', type=tuple, default=(100, 300),
                        help='Image dimension, i.e. width by channels for text')
    parser.add_argument('--epoch_steps', type=int, default=500,
                        help='Steps in an epoch')
    parser.add_argument('--val_steps', type=int, default=200,
                        help='Steps in an epoch')
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
    args = parser.parse_args()
    args.img_dim = (args.window_size, 300)

    print("Network configuration:")
    for key, value in parser.parse_args()._get_kwargs():
        print(key, value)

    run(args)
