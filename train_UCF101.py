import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl
import random
import time
from datetime import datetime
from video_config import VideoConfig

def train(config):
  model = Action_Recognizer(config)

  global_step = tf.Variable(0, trainable=False)

  logits, feat_map_placeholders = model.inference()

  loss, labels_placeholder = model.loss(logits)

  train_op = model.train(loss, global_step)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Create a saver.
  saver = tf.train.Saver()

  if not config.resume:
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
          log_device_placement=config.log_device_placement))
    sess.run(init)

  else:
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(config.train_dir)
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
  summary_writer = tf.train.SummaryWriter(config.train_dir,
                                          graph_def=graph_def)
  for step in xrange(config.max_steps):
    feat_maps_batch, labels = model.get_batch()
    dict = {i: d for i, d in zip(feat_map_placeholders, feat_maps_batch)}
    dict[labels_placeholder] = labels

    print ("epoch %d" % step)

    start_time = time.time()
    _, loss_value = sess.run([train_op, loss], feed_dict=dict)
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    # if step % 10 == 0:
    num_examples_per_step = config.batch_size_train
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                  'sec/batch)')
    print (format_str % (datetime.now(), step, loss_value,
                         examples_per_sec, sec_per_batch))

    # if step % 100 == 0:
    summary_str = sess.run(summary_op, feed_dict=dict)
    summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    # if step % 1000 == 0 or (step + 1) == config.max_steps:
    checkpoint_path = os.path.join(config.train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)


def main(_):
  config = VideoConfig()
  if not config.resume:
    if tf.gfile.Exists(config.train_dir):
      tf.gfile.DeleteRecursively(config.train_dir)
    tf.gfile.MakeDirs(config.train_dir)
  train(config)

if __name__ == '__main__':
  tf.app.run()