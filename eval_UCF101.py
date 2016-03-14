import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl
import random
import time
from datetime import datetime
import math
from video_config import VideoConfig

def eval_once(config, summary_writer, summary_op, model):
  # Calculate predictions.
  logits, feat_map_placeholders = model.inference(test=True)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(config.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Model restored.")

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    num_iter = int(math.ceil(config.nr_test_examples / config.batch_size_test))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * config.batch_size_test
    step = 0


    while step < num_iter:
      feat_maps_batch_segments, labels = model.get_test_batch()

      for segment in feat_maps_batch_segments:
        true_count_per_segment = 0
        dict = {i: d for i, d in zip(feat_map_placeholders, segment)}

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        predictions = sess.run([top_k_op], feed_dict=dict)
        true_count_per_segment += np.sum(predictions)

      true_count_per_segment = true_count_per_segment / len(feat_maps_batch_segments)
      step += 1
      true_count += true_count_per_segment

    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary_writer.add_summary(summary, global_step)


def evaluate(config):
  model = Action_Recognizer(config)

  # Restore the moving average version of the learned variables for eval.
  # variable_averages = tf.train.ExponentialMovingAverage(
  #     FLAGS.moving_average_decay)
  # variables_to_restore = variable_averages.variables_to_restore()
  # saver = tf.train.Saver(variables_to_restore)


  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
  summary_writer = tf.train.SummaryWriter(config.eval_dir,
                                          graph_def=graph_def)

  while True:
    eval_once(config, summary_writer, summary_op, model)
    if config.run_once:
      return
    time.sleep(config.eval_interval_secs)



def main(_):
  config = VideoConfig()
  if tf.gfile.Exists(config.eval_dir):
    tf.gfile.DeleteRecursively(config.eval_dir)
  tf.gfile.MakeDirs(config.eval_dir)
  evaluate(config)


if __name__ == '__main__':
  tf.app.run()