import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from model_MSVD import Video_Caption_Generator
from utils import load_pkl
from video_config import ActionConfig, CaptionConfig
import time
from datetime import datetime

def create_model(sess, saver, config_caption):
  """Create translation model and initialize or load parameters in session."""
  model = Video_Caption_Generator(config_caption)

  if config_caption.resume:
    ckpt = tf.train.get_checkpoint_state(config_caption.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      sess.run(tf.initialize_all_variables())
  else:
    print("Created model with fresh parameters.")
    sess.run(tf.initialize_all_variables())

  return model

def train(config_action, config_caption):

  model_action = Action_Recognizer(config_action)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
  summary_writer = tf.train.SummaryWriter(config_caption.eval_dir,
                                          graph_def=graph_def)

  # Calculate predictions.
  segments = []
  segment_placeholders = []
  for segment in config_caption.train_segments:
    output_caption, feat_map_placeholders = model_action.inference(caption=True)
    segments.append(segment)
    segment_placeholders.append(feat_map_placeholders)

  with tf.Session() as sess:
    # Create a saver.
    saver = tf.train.Saver()

    model = create_model(sess, saver, config_caption)

    # Calculate predictions.
    output, caption_placeholder = model.inference(segments)
    # tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(config_caption.train_dir,
                                          graph_def=graph_def)

    # Read data into buckets and compute their sizes.
    for step in xrange(config_caption.max_steps):
      feats_segments, caption = model.get_example()
      input_feed = {i: d for i, d in zip(segment_placeholders, feats_segments)}
      input_feed.update({i: d for i, d in zip(caption_placeholder, caption)})

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
  config_action = ActionConfig()
  config_caption = CaptionConfig()
  train(config_action, config_caption)

if __name__ == '__main__':
  tf.app.run()