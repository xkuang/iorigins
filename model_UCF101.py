from rcn_cell import GRCUCell, StackedGRCUCell
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from utils import load_pkl
from vgg import VGG
import cv2

class Action_Recognizer():
  def __init__(self, config, stacked=False):
    self._config = config
    self._kernel_size = 3
    self._cnn = VGG(self._config.nr_feat_maps, self._config.tensor_names, self._config.image_size)
    self._grcu_list = []

    for L, hidden_layer_size in enumerate(self._config.hidden_sizes):
      kernel = self._kernel_size if self._config.input_sizes[L][0] > self._kernel_size else self._config.input_sizes[L][0]
      if self._config.stacked:
        if L == 0:
          self._grcu_list.append(StackedGRCUCell(hidden_layer_size, -1, self._config.input_sizes[L][0],
                                                self._config.input_sizes[L][1], self._config.input_sizes[L][2], kernel))
        else:
          self._grcu_list.append(StackedGRCUCell(hidden_layer_size, self._config.hidden_sizes[L-1],
                                                self._config.input_sizes[L][0], self._config.input_sizes[L][1],
                                                self._config.input_sizes[L][2], kernel))
      else:
        self._grcu_list.append(GRCUCell(hidden_layer_size, self._config.input_sizes[L][0], self._config.input_sizes[L][1],
                                        self._config.input_sizes[L][2], kernel))


  def shuffle_train_data(self, train_data):
    index = list(train_data.index)
    np.random.shuffle(index)
    train_data = train_data.ix[index]

    return train_data


  def get_video_data(self):
    video_data = pd.read_csv(self._config.video_data_path, sep=',')
    video_data['feat_path'] = video_data['feat_path'].map(lambda x: os.path.join(self._config.feats_dir, x))
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(self._config.videos_dir, x))
    video_data = video_data[video_data['feat_path'].map(lambda x: os.path.exists( x ))]

    return video_data


  def get_test_data(self):
    video_data = pd.read_csv(self._config.test_data_path, sep=',')
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(self._config.videos_dir, x))

    return video_data


  def get_batch(self):
    train_data = self.get_video_data()
    nr_training_examples = train_data.shape[0]
    train_data = self.shuffle_train_data(train_data)

    current_batch_indices = random.sample(xrange(nr_training_examples), self._config.batch_size_train)
    current_batch = train_data.ix[current_batch_indices]

    current_videos = current_batch['video_path'].values
    current_feats = current_batch['feat_path'].values
    labels = current_batch['label'].values

    current_feats_vals = map(lambda vid: load_pkl(vid), current_feats)
    feat_maps_batch =  zip(*current_feats_vals)

    feat_maps_batch = map(lambda x: np.asarray(x), feat_maps_batch)
    shape = feat_maps_batch[4].shape
    feat_maps_batch[4] = np.reshape(feat_maps_batch[4], [shape[0], shape[1], 1, 1, shape[2]])

    return feat_maps_batch, np.asarray(labels)


  def get_test_batch(self):
    test_data = self.get_test_data()
    nr_testing_examples = test_data.shape[0]
    current_batch_indices = random.sample(xrange(nr_testing_examples), self._config.batch_size_test)
    current_batch = test_data.ix[current_batch_indices]

    current_videos = current_batch['video_path'].values
    labels = current_batch['label'].values

    current_feats_vals = map(lambda vid: self.get_test_features(vid), current_videos)
    feat_maps_batch_segments = zip(*current_feats_vals)

    feat_maps_batch_segments = map(lambda segment: zip(*segment), feat_maps_batch_segments)
    feat_maps_batch_segments = map(lambda segment: map(lambda feat: np.asarray(feat), segment), feat_maps_batch_segments)
    # feat_maps_batch_segments = map( lambda batch: map(lambda segment: map(lambda feat_map: np.asarray(feat_map), segment),
    #                                feat_maps_batch_segments)

    return feat_maps_batch_segments, np.asarray(labels)


  def get_test_features(self, vid):
    try:
      cap = cv2.VideoCapture(vid)
    except:
      print ("Cant open video capture")

    frame_count = 0
    frame_list = []

    while True:
      # Capture frame-by-frame
      ret, frame = cap.read()

      if ret is False:
          break

      frame_list.append(frame)
      frame_count += 1

    if frame_count == 0:
      print "This video could not be processed"
      return

    frame_list = np.array(frame_list)

    if frame_count < self._config.test_segments * self._config.nr_frames:
      print ("This video is too short. It has %d frames" % frame_count)

    segment_indices = np.linspace(0, frame_count, num=self._config.test_segments, endpoint=False).astype(int)

    segment_list = []
    for segment_idx in segment_indices:
      segment_frames = frame_list[segment_idx : (segment_idx + self._config.nr_frames)]
      cropped_segment_frames = np.array(map(lambda x: self._cnn.preprocess_frame(self._config.cropping_sizes, x), segment_frames))
      segment_feats = self._cnn.get_features(cropped_segment_frames)
      shape = segment_feats[4].shape
      segment_feats[4] = np.reshape(segment_feats[4], [shape[0], 1, 1, shape[1]])
      segment_list.append(segment_feats)

    return segment_list


  def inference(self, test=False):
    if test:
      batch_size = self._config.batch_size_test
    else:
      batch_size = self._config.batch_size_train


    # feature map placeholders
    feat_map_placeholders = []
    for i, input_size in enumerate(self._config.input_sizes):
      feat_map_placeholders.append(tf.placeholder(tf.float32, [batch_size,
                                                               self._config.nr_frames,
                                                               input_size[0],
                                                               input_size[1],
                                                               input_size[2]], name=("feat_map_%d" % i)))

    states = []
    for grcu in self._grcu_list:
      states.append(grcu.zero_state(batch_size))

    with tf.variable_scope("RNN"):
      for L, grcu in enumerate(self._grcu_list):
        with tf.variable_scope('GRU-L%d' % L):
          for time_step in range(self._config.nr_frames):
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            if self._config.stacked:
              if L == 0:
                states[L] = grcu(tf.convert_to_tensor(feat_map_placeholders[L][:,time_step,:,:,:]), states[L], None)
              else:
                states[L] = grcu(tf.convert_to_tensor(feat_map_placeholders[L][:,time_step,:,:,:]), states[L], states[L-1])
            else:
              states[L] = grcu(tf.convert_to_tensor(feat_map_placeholders[L][:,time_step,:,:,:]), states[L])

            self.activation_summary(states[L])


    for L in range(len(self._grcu_list)):
      final_state = states[L]

      with tf.variable_scope("softmax-L%d" % L) as scope:
        avg_pool = tf.nn.avg_pool(final_state,
                                  ksize=[1, self._config.input_sizes[L][0], self._config.input_sizes[L][1], 1],
                                  strides=[1, 1, 1, 1], padding='VALID', name=('avg_pool-L%d' % L))
        dropout = tf.nn.dropout(avg_pool, self._config.keep_prob)

        dropout_shape = dropout.get_shape()
        reshaped_output = tf.reshape(dropout, [dropout_shape[0].value, dropout_shape[-1].value])

        weights_soft = tf.get_variable("weights", shape=[self._config.hidden_sizes[L], self._config.nr_classes],
                        initializer=tf.truncated_normal_initializer(stddev=0.04), dtype=tf.float32)
        biases_soft = tf.get_variable("biases", shape=[self._config.nr_classes],
                        initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        softmax = tf.nn.xw_plus_b(reshaped_output, weights_soft, biases_soft, name=scope.name)
        tf.add_to_collection('predictions_ensamble', softmax)

      self.activation_summary(softmax)

      y = tf.constant(len(tf.get_collection('predictions_ensamble')), dtype=tf.float32)
    return tf.truediv(tf.add_n(tf.get_collection('predictions_ensamble'), name='softmax_linear_sum'),
                      y, name='softmax_linear_average'), feat_map_placeholders


  def loss(self, logits):
    labels_placeholder = tf.placeholder(tf.int64, [self._config.batch_size_train], name="labels")
    # dense_labels = self.sparse_to_dense(labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels_placeholder, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss, labels_placeholder


  def train(self, total_loss, global_step):

    tf.scalar_summary('learning_rate', self._config.learning_rate)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = self.add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(self._config.learning_rate)
      # opt = tf.train.AdamOptimizer(conv_config.INITIAL_LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)
      grads_and_vars = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradients_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    self.add_histograms(grads_and_vars)

    variables_averages_op = self.add_moving_averages_to_all(global_step)

    with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op


  def variable_with_weight_decay(self, name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = self.variable(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
      tf.add_to_collection('losses', weight_decay)
    return var


  def variable(self, name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    # with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

  def add_loss_summaries(self, total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.scalar_summary(l.op.name + ' (raw)', l)
      tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

  def activation_summary(self, x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

  def kernel_image_summary(self, kernel):
    kernel_reversed = tf.transpose(kernel, [3, 0, 1, 2])
    for i in xrange(10):
            tf.image_summary(("images/filter%d" % (i)), kernel_reversed)


  def add_moving_averages_to_all(self, global_step):
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            self._config.moving_average_decay, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op


  def add_histograms(self, grads_and_vars):
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads_and_vars:
      if grad:
        tf.histogram_summary(var.op.name + '/gradients', grad)