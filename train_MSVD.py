import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_MSVD import Video_Caption_Generator
from utils import load_pkl

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_MSVD',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('video_data_path', './data/video_corpus.csv',
                           """path to video corpus""")

tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/MSVD',
                           """youtube clips path""")

tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats_MSVD',
                           """youtube features path""")

tf.app.flags.DEFINE_string('index_to_word_path', '/media/ioana/Elements/index_to_word_MSVD/dict.npy',
                           """index_to_word dictionary path""")

tf.app.flags.DEFINE_string('input_sizes',  [[56, 56, 128],
                                            [28, 28, 256],
                                            [14, 14, 512],
                                            [7,  7, 512],
                                            [1, 1, 4096]],
                           """the size of the input image/frame""")
tf.app.flags.DEFINE_string('hidden_sizes', [64, 128, 256, 256, 512],
                           """youtube features path""")
tf.app.flags.DEFINE_string('batch_size_train', 16,
                           """Nr of batches""")
tf.app.flags.DEFINE_string('nr_frames', 10,
                           """Nr of sample frames at equally-space segments.""")
tf.app.flags.DEFINE_string('nr_segments', 5,
                           """Nr of segments to sample frames from.""")
tf.app.flags.DEFINE_string('nr_feat_maps', 5,
                           """Nr of feature maps extracted from the inception CNN for each frame.""")
tf.app.flags.DEFINE_string('max_steps', 1000,
                           """Nr of epochs to train.""")
tf.app.flags.DEFINE_string('learning_rate', 0.001,
                           """Model's learning rate.""")

def train():

  model = Video_Caption_Generator(
            video_data_path=FLAGS.video_data_path,
            feats_dir=FLAGS.feats_dir,
            videos_dir=FLAGS.videos_dir,
            index_to_word_path=FLAGS.index_to_word_path,
            input_sizes=FLAGS.input_sizes,
            hidden_sizes=FLAGS.hidden_sizes,
            batch_size_train=FLAGS.batch_size_train,
            nr_frames=FLAGS.nr_frames,
            nr_feat_maps=FLAGS.nr_feat_maps,
            nr_classes=FLAGS.nr_classes,
            keep_prob=FLAGS.keep_prob,
            nr_epochs_per_decay=FLAGS.nr_epochs_per_decay,
            moving_average_decay=FLAGS.moving_average_decay,
            initial_learning_rate=FLAGS.learning_rate,
            learning_rate_decay_factor=FLAGS.learning_rate_decay_factor)


def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()