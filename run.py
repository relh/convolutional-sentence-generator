from __future__ import print_function

from collections import defaultdict
import os
import math
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

import marisa_trie

from fastText import load_model
from fastText import tokenize

import densenet

mode = 'char'
version = '3v6'
name = 'models/' + version
word_folder = 'data/billion_word/training-monolingual.tokenized.shuffled/'
eye = np.eye(190)
final_probs = np.zeros((1, 10000))#len(counts.keys()))

# val_perplexity
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=name+'.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=0.00001, verbose=1)


def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity

total = 0
def trie_recurse(wordinds, charinds, prefix, probs, cum, trie, model, new_inp):
  global total
  total += 1
  num = 0
  for let in charinds.keys():
    #for batch_i in range(128): #args.batch_size:
    new_inp[0][-1] = eye[charinds[let]]#preds[0] # change back to 1dx
    keys = trie.keys(prefix+let)
    num = len(trie.keys(prefix+let))
    if num == 1:
        final_probs[0][wordinds[keys[0]]] = np.multiply(cum, probs[0][charinds[let]])
    elif num > 1:
        probs = model.predict(new_inp)
        new_inp = np.roll(new_inp, -1, 1) # change back to 1dx
      
        cum = np.multiply(cum, probs[0][charinds[let]])
        trie_recurse(wordinds, charinds, prefix+let, probs, cum, trie, model, new_inp)
  print("{} / {}. Total".format(total, 8500))

def check_pred(model, indices, inp):
    preds = model.predict(inp)#, label)
    print(np.argmax(preds))
    for k,v in indices.items():
      if v ==  np.argmax(preds):
        print(k)
        new_inp = np.roll(inp, -1, 1) # change back to 1dx
        new_inp[0][-1] = eye[indices[k]]#preds[0] # change back to 1dx
    return new_inp


def char_to_bpc(model, timeseries, wordinds, indices, words, args, trie):
    pass


def char_to_perplexity(model, timeseries, wordinds, indices, words, args, trie):
    accum = 0
    words_len = len(words)-args.window_size
    batches = math.floor(words_len / args.batch_size)
    print(batches)
    for start in range(0, 1): # 500batches):
      #idx = start*args.batch_size
      inp = np.array([timeseries[i:i+args.window_size] for i in range(start+args.batch_size, start+args.batch_size+1)])
      label = np.asarray([eye[indices[x]] for x in words[start+args.window_size+args.batch_size:start+args.window_size+args.batch_size+1]])
      print([x for x in words[start+args.window_size+args.batch_size:start+args.window_size+args.batch_size+1]])
      
      preds = model.predict(inp)#, label)
      new_inp = np.roll(inp, -1, 1) # change back to 1dx
      #new_inp = np.expand_dims(inp[0], 0)
      #new_inp = check_pred(model, indices, inp)
      #new_inp = check_pred(model, indices, new_inp)
      #new_inp = check_pred(model, indices, new_inp)
      #new_inp = check_pred(model, indices, new_inp)

      # Calc softmax for all words
      cum = np.ones(1)
      #trie_recurse(wordinds, indices, '', preds, cum, trie, model, new_inp)

      final_probie = np.load('final_probs.npy')
      final_probie = final_probie / sum(final_probie[0])
      print(sum(final_probie[0]))
      print(wordinds['the'])
      print(np.exp(-np.log(final_probie[0][wordinds['then']])))
      print(final_probie[0][wordinds['the']])
      print(final_probie[0][wordinds['than']])
      print(np.argmax(final_probie[0]))
      for k,v in wordinds.items():
        if v == np.argmax(final_probie[0]):
          print(k)
      print(final_probie[0][0])
      """

      np_vecs = np.asarray(final_probs)#, dtype=np.int8)
      print(np_vecs.shape)
      print(label.shape)
      np.save('final_probs', np_vecs)
      np.save('final_labels', label)
      """


def word_to_perplexity(model, timeseries, indices, words, args):
    accum = 0
    words_len = len(words)-args.window_size
    batches = math.floor(words_len / args.batch_size)
    print(batches)
    for start in range(0, batches):
      idx = start*args.batch_size
      inp = np.array([timeseries[i:i+args.window_size] for i in range(idx, idx+args.batch_size)])
      label = np.asarray([indices[x] for x in words[idx+args.window_size:idx+args.window_size+args.batch_size]])  
      
      pred = model.predict(inp, batch_size=128)
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


def load_trie(counts):
    if os.path.exists('words_trie.marisa'):
      trie = marisa_trie.Trie()
      trie.load('words_trie.marisa')
    else:
      trie = marisa_trie.Trie(counts.keys())
      trie.save('words_trie.marisa')
    return trie


def load_input(path):
    counts = defaultdict(int)
    if not os.path.exists(mode+'indices.p'):
      root = '/'.join(path.split('/')[0:-1])
      all_paths = [root+'/'+x for x in os.listdir(root)] #'/'.join(path.split('/')[0:-1]))
    else:
      all_paths = [path]
    
    for path in all_paths:
      print(path)
      with open(path) as f:
        if mode == 'word':
          words = tokenize(f.read())
        else:
          words = f.read()

        for word in words:
          counts[word] += 1 

    words = [x for x in words if len(x) > 0]
    return words, counts


def load_indices(mode='char', words=None, counts=None):
    if os.path.exists(mode+'indices.p'):
      indices = pickle.load(open(mode+'indices.p', 'rb'), encoding='latin1')
    else:
      indices = {}
      i = 0
      for word in counts.keys():
        indices[word] = int(i)
        indices[i] = str(word)
        i += 1
      print("i is: " + str(i))
      print("len is: " + str(len(indices.keys())))
      pickle.dump(indices, open(mode+'indices.p', 'wb'))
    return indices


def make_embedding(path, words, indices):
    #root = '/'.join(path.split('/')[0:-1])
    #all_paths = [root+'/'+x for x in os.listdir(root)] #'/'.join(path.split('/')[0:-1]))
    #for path in all_paths:
    vec_path = 'data/'+path.split('/')[-1]+'_'+mode
    print(vec_path)
    if os.path.exists(vec_path+'.npy'):
      np_vecs = np.load(vec_path+'.npy')
    else:
      words_len = len(words)
      vecs = []
      if mode == 'word':
        f = load_model('wiki.en.bin')
      for i, w in enumerate(words):
        if mode == 'word':
          vec = f.get_word_vector(w)
        else:
          vec = eye[indices[w]]
        vecs.append(vec) 
        if i % 10000 == 0:
          print("{} / {}".format(i, words_len))
      np_vecs = np.asarray(vecs, dtype=np.int8)
      np.save(vec_path, np_vecs)
    return np_vecs


def run(args):
    ####################
    # Data Processing  #
    ####################

    words, counts = load_input(args.train_path)
    trie = marisa_trie.Trie(counts.keys())
    indices = load_indices('word', words, counts)
    args.nb_classes = len(indices.keys())
    print(len(indices.keys()))
    timeseries = make_embedding(args.train_path, words, indices)

    ###################
    # Construct model #
    ###################

    if os.path.exists(name+'.h5'):
      model = keras_load_model(name+'.h5')#', custom_objects={'perplexity': perplexity})
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
                    metrics=['mae', 'accuracy'])

    ####################
    # Network training #
    ####################

    #train(model, timeseries, indices, words, args)

    #words, counts = load_input(args.test_path)
    #timeseries = make_embedding(args.test_path, words, indices)
    trie = load_trie(counts)
    charinds = load_indices('char')
    char_to_perplexity(model, timeseries, indices, charinds, words, args, trie) #cel
    #word_to_perplexity(model, timeseries, indices, words, args) #nll
    #inp = np.array([timeseries[i:i+args.window_size] for i in range(idx, idx+args.batch_size)])
    #model.predict(inp)
    


def generator(timeseries, indices, words, args, bot=1, top=-1):
    fil = 0
    if top < 0:
      top = len(timeseries)
    #start = bot
    start = random.choice(range(bot, top)) #len(timeseries)-window_size,1)
    while True:
        # choose random index in features
        start += args.window_size+args.batch_size
        if start >= (top-args.batch_size-args.window_size):
          start = bot
        batch_features = np.array([timeseries[i:i+args.window_size] for i in range(start, start+args.batch_size)])
        if mode == 'word':
          batch_labels = np_utils.to_categorical(np.asarray([indices[x] for x in words[start+args.window_size:start+args.window_size+args.batch_size]]), args.nb_classes)
        else:
          batch_labels = np.asarray([eye[indices[x]] for x in words[start+args.window_size:start+args.window_size+args.batch_size]])

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
    model.fit_generator(generator(timeseries, indices, words, args, 1, top), steps_per_epoch=args.epoch_steps, epochs=args.nb_epoch, validation_data=generator(timeseries, indices, words, args, top, len(timeseries)-args.window_size-args.batch_size), validation_steps=args.val_steps, callbacks=[lr_reducer, checkpointer], shuffle=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NLP experiment')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=25000, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=22,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=512,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=128,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')
    parser.add_argument('--test_path', type=str, default='data/test.txt',
                        help='Specify file path to train on')
    parser.add_argument('--train_path', type=str, default='data/train.txt',
                        help='Specify file path to test on')
    parser.add_argument('--window_size', type=int, default=500,
                        help='How many words to use as context')
    parser.add_argument('--nb_classes', type=int, default=10000,
                        help='Number of classes')
    parser.add_argument('--img_dim', type=tuple, default=(500, 190),
                        help='Image dimension, i.e. width by channels for text')
    parser.add_argument('--epoch_steps', type=int, default=1000,
                        help='Steps in an epoch')
    parser.add_argument('--val_steps', type=int, default=300,
                        help='Steps in an epoch')
    """ Run Conv NLP experiments 
    billion_word/training-monolingual.tokenized.shuffled/news.en-00001-of-00100

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
    args.img_dim = (args.window_size, args.img_dim[1])
    #args.train_path = random.choice(os.listdir(word_folder))
    #args.test_path = random.choice(os.listdir(word_folder))

    print("Network configuration:")
    for key, value in parser.parse_args()._get_kwargs():
        print(key, value)

    run(args)
