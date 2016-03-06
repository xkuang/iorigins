from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import cPickle as pickle
import tensorflow as tf
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './youtube2text',
                           """Absolute path to data dir.""")

tf.app.flags.DEFINE_string('train_pickle', 'train.pkl',
                           """Absolute path to train data.""")

tf.app.flags.DEFINE_string('valid_pickle', 'valid.pkl',
                           """Absolute path to validation data.""")

tf.app.flags.DEFINE_string('test_pickle', 'test.pkl',
                           """Absolute path to test data.""")

tf.app.flags.DEFINE_string('CAP_pickle', 'CAP.pkl',
                           """Absolute path to test data.""")

tf.app.flags.DEFINE_string('FEAT_pickle', 'FEAT_key_vidID_value_features.pkl',
                           """Absolute path to test data.""")

tf.app.flags.DEFINE_string('worddict_pickle', 'worddict.pkl',
                           """Absolute path to test data.""")

class Video2text(object):

  def __init__(self, batch_size_train, batch_size_valid, batch_size_test):
    self.batch_size_train = batch_size_train
    self.batch_size_valid = batch_size_valid
    self.batch_size_test = batch_size_test
    self.load_data()

  def load_data(self):
    print ('loading youtube2text features')

    with open(os.path.join(FLAGS.data_dir, FLAGS.train_pickle), 'rb') as f:
      self.train = pickle.load(f)

    with open(os.path.join(FLAGS.data_dir, FLAGS.test_pickle), 'rb') as f:
      self.test = pickle.load(f)

    with open(os.path.join(FLAGS.data_dir, FLAGS.valid_pickle), 'rb') as f:
      self.valid = pickle.load(f)

    with open(os.path.join(FLAGS.data_dir, FLAGS.CAP_pickle), 'rb') as f:
      self.cap = pickle.load(f)

    with open(os.path.join(FLAGS.data_dir, FLAGS.FEAT_pickle), 'rb') as f:
      self.feat = pickle.load(f)

    with open(os.path.join(FLAGS.data_dir, FLAGS.worddict_pickle), 'rb') as f:
      self.worddict = pickle.load(f)

    self.rev_worddict = dict()
    # wordict start with index 2
    for k, v in self.worddict.iteritems():
        self.rev_worddict[v] = k
    self.rev_worddict[0] = '<eos>'
    self.rev_worddict[1] = 'UNK'

    self.train_batches = create_batches(len(self.train), self.batch_size_train)
    self.valid_batches = create_batches(len(self.valid), self.batch_size_valid)
    self.test_batches = create_batches(len(self.test), self.batch_size_test)




