import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from model_MSVD import Video_Caption_Generator
from utils import load_pkl
from video_config import ActionConfig, CaptionConfig

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

    train_data, _ = model.get_video_data()
    word_to_index, index_to_word, bias_init_vector = model.get_caption_dicts(train_data)

    # Calculate predictions.
    output, caption_placeholders = model.inference(segments)

    # Read data into buckets and compute their sizes.


def main(_):
  config_action = ActionConfig()
  config_caption = CaptionConfig()
  train(config_action, config_caption)

if __name__ == '__main__':
  tf.app.run()