"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import re
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

class GRCUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Convolutional Unit cell
  (cf. http://arxiv.org/pdf/1511.06432v4).
  """

  def __init__(self, hidden_size, input_width, input_height, input_size, kernel_size):
    """Initialize the parameters for an GRCU cell.
    Args:
      hidden_dim: int, The number of dimensions in the GRCU cell
      input_width: int, The width of the input map
      input_height: int, The height of the input map
      input_dim: int, The dimensionality of the inputs into the GRCU cell
    """

    self._hidden_size = hidden_size
    self._input_width = input_width
    self._input_height = input_height
    self._input_size = input_size
    self._kernel_size = kernel_size

    dtype = tf.float32
    initializer = tf.truncated_normal_initializer(stddev=1e-4)
    #update gate kernels
    with tf.variable_scope("Gates"):
      self.W_z = tf.get_variable(
            "W_z", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U_z = tf.get_variable(
            "U_z", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

      #reset gate kernels
      self.W_r = tf.get_variable(
            "W_r", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U_r = tf.get_variable(
            "U_r", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

    #candidate gate kernels
    with tf.variable_scope("Candidate"):
      self.W = tf.get_variable(
                "W", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U = tf.get_variable(
            "U", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

  @property
  def input_width(self):
    return self._input_width

  @property
  def input_height(self):
    return self._input_height

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._hidden_size

  @property
  def state_size(self):
    return [self._input_width, self._input_height, self._hidden_size]

  @property
  def kernel_size(self):
    return self._kernel_size

  def __call__(self, input_, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""

    #convolution operations for update gate
    conv_W_z = tf.nn.conv2d(input_, self.W_z, [1, 1, 1, 1], padding="SAME")
    # activation_summary(conv_W_z)
    conv_U_z = tf.nn.conv2d(state, self.U_z, [1, 1, 1, 1], padding="SAME")

    u = conv_W_z + conv_U_z
    u = sigmoid(u)

    #convolution operations for reset gate
    conv_W_r = tf.nn.conv2d(input_, self.W_r, [1, 1, 1, 1], padding="SAME")
    conv_U_r = tf.nn.conv2d(state, self.U_r, [1, 1, 1, 1], padding="SAME")

    r = conv_W_r + conv_U_r
    r = sigmoid(r)

    #convolution operations for candidate gate
    conv_W = tf.nn.conv2d(input_, self.W, [1, 1, 1, 1], padding="SAME")
    conv_U = tf.nn.conv2d(r * state, self.U, [1, 1, 1, 1], padding="SAME")

    c = conv_W + conv_U
    c = tanh(c)

    new_h = u * state + (1 - u) * c

    return new_h, new_h

  # def _activation_summary(self, x):
  #   """Helper to create summaries for activations.
  #
  #   Creates a summary that provides a histogram of activations.
  #   Creates a summary that measure the sparsity of activations.
  #
  #   Args:
  #     x: Tensor
  #   Returns:
  #     nothing
  #   """
  #   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  #   # session. This helps the clarity of presentation on tensorboard.
  #   # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  #   tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  #   tf.histogram_summary(tensor_name + '/activations', x)
  #   tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

class StackedGRCUCell(tf.nn.rnn_cell.RNNCell):
  """Stacked Gated Recurrent Convolutional Unit cell
  (cf. http://arxiv.org/pdf/1511.06432v4).
  """

  def __init__(self, hidden_size, input_width, input_height, input_size, kernel_size):
    """Initialize the parameters for an GRCU cell.
    Args:
      hidden_dim: int, The number of dimensions in the GRCU cell
      input_width: int, The width of the input map
      input_height: int, The height of the input map
      input_dim: int, The dimensionality of the inputs into the GRCU cell
    """

    self._hidden_size = hidden_size
    self._input_width = input_width
    self._input_height = input_height
    self._input_size = input_size
    self._kernel_size = kernel_size

    dtype = tf.float32
    initializer = tf.truncated_normal_initializer(stddev=1e-4)
    #update gate kernels
    with tf.variable_scope("Gates"):
      self.W_z = tf.get_variable(
            "W_z", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U_z = tf.get_variable(
            "U_z", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

      #reset gate kernels
      self.W_r = tf.get_variable(
            "W_r", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U_r = tf.get_variable(
            "U_r", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

    #candidate gate kernels
    with tf.variable_scope("Candidate"):
      self.W = tf.get_variable(
                "W", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size],
        initializer=initializer, dtype=dtype)
      self.U = tf.get_variable(
            "U", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size],
        initializer=initializer, dtype=dtype)

  @property
  def input_width(self):
    return self._input_width

  @property
  def input_height(self):
    return self._input_height

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._hidden_size

  @property
  def state_size(self):
    return [self._input_width, self._input_height, self._hidden_size]

  @property
  def kernel_size(self):
    return self._kernel_size

  def __call__(self, input_, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""

    #convolution operations for update gate
    conv_W_z = tf.nn.conv2d(input_, self.W_z, [1, 1, 1, 1], padding="SAME")
    # activation_summary(conv_W_z)
    conv_U_z = tf.nn.conv2d(state, self.U_z, [1, 1, 1, 1], padding="SAME")

    u = conv_W_z + conv_U_z
    u = sigmoid(u)

    #convolution operations for reset gate
    conv_W_r = tf.nn.conv2d(input_, self.W_r, [1, 1, 1, 1], padding="SAME")
    conv_U_r = tf.nn.conv2d(state, self.U_r, [1, 1, 1, 1], padding="SAME")

    r = conv_W_r + conv_U_r
    r = sigmoid(r)

    #convolution operations for candidate gate
    conv_W = tf.nn.conv2d(input_, self.W, [1, 1, 1, 1], padding="SAME")
    conv_U = tf.nn.conv2d(r * state, self.U, [1, 1, 1, 1], padding="SAME")

    c = conv_W + conv_U
    c = tanh(c)

    new_h = u * state + (1 - u) * c

    return new_h, new_h

  # def _activation_summary(self, x):
  #   """Helper to create summaries for activations.
  #
  #   Creates a summary that provides a histogram of activations.
  #   Creates a summary that measure the sparsity of activations.
  #
  #   Args:
  #     x: Tensor
  #   Returns:
  #     nothing
  #   """
  #   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  #   # session. This helps the clarity of presentation on tensorboard.
  #   # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  #   tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  #   tf.histogram_summary(tensor_name + '/activations', x)
  #   tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
