import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl
import random
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_UCF101',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('video_data_path', './ucfTrainTestlist/train_data.csv',
                           """path to video corpus""")

# tf.app.flags.DEFINE_string('videos_dir', './UCF-101',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/7ED0-6463/UCF-101',
                           """youtube clips path""")

# tf.app.flags.DEFINE_string('feats_dir', './feats_ucf',
#                            """youtube features path""")
tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/7ED0-6463/feats_ucf',
                           """youtube features path""")

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
                           """Nr of sample frames at equally-space intervals.""")
tf.app.flags.DEFINE_string('nr_classes', 101,
                           """Nr of classes.""")
tf.app.flags.DEFINE_string('nr_feat_maps', 5,
                           """Nr of feature maps extracted from the inception CNN for each frame.""")
tf.app.flags.DEFINE_string('max_steps', 1000,
                           """Nr of epochs to train.""")
tf.app.flags.DEFINE_string('learning_rate', 0.001,
                           """Model's learning rate.""")
tf.app.flags.DEFINE_string('learning_rate_decay_factor', 0.6,
                           """Model's learning rate decay factor.""")
tf.app.flags.DEFINE_string('keep_prob', 0.7,
                           """Dropout ration for the last layer of the classifiers.""")
tf.app.flags.DEFINE_string('nr_epochs_per_decay', 350,
                           """Number of epochs per decay of the learning rate.""")
tf.app.flags.DEFINE_string('moving_average_decay', 0.9999,
                           """Moving average decay rate.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('resume', True,
                           """Variable to specify if the last model should be resumed or a new one created""")


def train(resume=False):
  model = Action_Recognizer(
          video_data_path=FLAGS.video_data_path,
          feats_dir=FLAGS.feats_dir,
          videos_dir=FLAGS.videos_dir,
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

  global_step = tf.Variable(0, trainable=False)

  logits, feat_map_placeholders = model.inference()

  loss, labels_placeholder = model.loss(logits)

  train_op = model.train(loss, global_step)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables())

  if not resume:

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

  else:
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint

        saver.restore(sess, ckpt.model_checkpoint_path)

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.

        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      else:
        print('No checkpoint file found. Cannot resume.')
        return

  graph_def = sess.graph.as_graph_def(add_shapes=True)
  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                          graph_def=graph_def)
  for step in xrange(FLAGS.max_steps):
    feat_maps_batch, labels = model.get_batch()
    dict = {i: d for i, d in zip(feat_map_placeholders, feat_maps_batch)}
    dict[labels_placeholder] = labels

    print ("epoch %d" % step)

    start_time = time.time()
    _, loss_value = sess.run([train_op, loss], feed_dict=dict)
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 10 == 0:
      num_examples_per_step = FLAGS.batch_size_train
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,
                           examples_per_sec, sec_per_batch))

    if step % 100 == 0:
      summary_str = sess.run(summary_op, feed_dict=dict)
      summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(_):
  if FLAGS.resume:
    train(FLAGS.resume)
  else:
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()