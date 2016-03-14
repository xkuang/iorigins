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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './train_UCF101',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_dir', './eval_UCF101',
                           """Directory where to write event logs for evaluation step"""
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('video_data_path', './ucfTrainTestlist/train_data.csv',
                           """path to video corpus""")
tf.app.flags.DEFINE_string('test_data_path', './ucfTrainTestlist/test_data.csv',
                           """path to video corpus""")
tf.app.flags.DEFINE_string('videos_dir', './UCF-101',
                           """youtube clips path""")
# tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/7ED0-6463/UCF-101',
#                            """youtube clips path""")

tf.app.flags.DEFINE_string('feats_dir', './feats_ucf',
                           """youtube features path""")
# tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/7ED0-6463/feats_ucf',
#                            """youtube features path""")

tf.app.flags.DEFINE_string('batch_size_test', 1,
                           """Nr of batches""")
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
tf.app.flags.DEFINE_string('nr_frames', 3,
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

tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_string('batch_size', 16,
                           """Nr of batches""")
tf.app.flags.DEFINE_string('image_size', 224,
                           """the size of the image that goes into the VGG net""")
tf.app.flags.DEFINE_string('tensor_names', ["import/pool2:0", "import/pool3:0", "import/pool4:0", "import/pool5:0", "import/Relu_1:0"],
                           """the names of the tensors to run in the vgg network""")
tf.app.flags.DEFINE_string('test_segments', 2,
                           """The number of segments to extract 10 frames to test from""")
tf.app.flags.DEFINE_string('cropping_sizes', [240, 224, 192, 168],
                           """the cropping sizes to randomly sample from""")

def eval_once(saver, summary_writer, summary_op, model):
  # Calculate predictions.
  logits, feat_map_placeholders = model.inference(test=True)

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
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


def evaluate():
  model = Action_Recognizer(
          video_data_path=FLAGS.video_data_path,
          test_data_path=FLAGS.test_data_path,
          feats_dir=FLAGS.feats_dir,
          videos_dir=FLAGS.videos_dir,
          input_sizes=FLAGS.input_sizes,
          hidden_sizes=FLAGS.hidden_sizes,
          batch_size_train=FLAGS.batch_size_train,
          batch_size_test=FLAGS.batch_size_test,
          nr_frames=FLAGS.nr_frames,
          nr_feat_maps=FLAGS.nr_feat_maps,
          nr_classes=FLAGS.nr_classes,
          keep_prob=FLAGS.keep_prob,
          moving_average_decay=FLAGS.moving_average_decay,
          initial_learning_rate=FLAGS.learning_rate,
          tensor_names=FLAGS.tensor_names,
          image_size=FLAGS.image_size,
          test_segments=FLAGS.test_segments,
          cropping_sizes=FLAGS.cropping_sizes)



  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
  summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                          graph_def=graph_def)

  while True:
    eval_once(saver, summary_writer, summary_op, model)
    if FLAGS.run_once:
      return
    time.sleep(FLAGS.eval_interval_secs)



def main(_):
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()