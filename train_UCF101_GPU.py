import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl
import random
import time
import datetime
import re

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_UCF101',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('video_data_path', './ucfTrainTestlist/train_data.csv',
                           """path to video corpus""")

tf.app.flags.DEFINE_string('videos_dir', './UCF-101',
                           """youtube clips path""")
# tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/7ED0-6463/UCF-101',
#                            """youtube clips path""")

tf.app.flags.DEFINE_string('feats_dir', './feats_ucf',
                           """youtube features path""")
# tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/7ED0-6463/feats_ucf',
#                            """youtube features path""")

tf.app.flags.DEFINE_string('input_sizes',  [[56, 56, 128],
                                            [28, 28, 256],
                                            [14, 14, 512],
                                            [7,  7, 512],
                                            [1, 1, 4096]],
                           """the size of the input image/frame""")
tf.app.flags.DEFINE_string('hidden_sizes', [64, 128, 256, 256, 512],
                           """youtube features path""")
tf.app.flags.DEFINE_string('batch_size_train', 64,
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
tf.app.flags.DEFINE_boolean('tower_name', 'tower',
                            """Tower name for gpus.""")
tf.app.flags.DEFINE_boolean('num_gpus', 1,
                            """Number of gpus.""")
tf.app.flags.DEFINE_boolean('gpu', True,
                            """Bool var that indicates weather it should run on cpu or gpu.""")

def tower_loss(scope):
  """Calculate the total loss on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

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

  feat_map_batch, labels = model.get_batch()

  # Build inference Graph.
  logits = model.inference(feat_map_batch)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  loss = model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():

  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    lr = FLAGS.learning_rate
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (FLAGS.tower_name, i)) as scope:
           # Calculate the loss for one tower of the model. This function
          # constructs the entire model but shares the variables across
          # all towers.
          loss = tower_loss(scope)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad:
        summaries.append(
            tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=graph_def)

    for step in xrange(FLAGS.max_steps):
      print ("epoch %d" % step)
      # dict = {}
      # for i, input_size in enumerate(FLAGS.input_sizes):
      #   dict["feat_map_%d" % i] = feat_maps_batch[i]

      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
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
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


def main(_):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()