"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
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
    dtype = input_.dtype

    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        #update gate kernels
        W_z = vs.get_variable(
              "W_z", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size], dtype=dtype)
        U_z = vs.get_variable(
              "U_z", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size], dtype=dtype)

        #convolution operations for update gate
        conv_W_z = tf.nn.conv2d(input_, W_z, [1, 1, 1, 1], padding="SAME")
        conv_U_z = tf.nn.conv2d(state, U_z, [1, 1, 1, 1], padding="SAME")

        u = conv_W_z + conv_U_z
        u = sigmoid(u)

        #reset gate kernels
        W_r = vs.get_variable(
              "W_r", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size], dtype=dtype)
        U_r = vs.get_variable(
              "U_r", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size], dtype=dtype)

        #convolution operations for reset gate
        conv_W_r = tf.nn.conv2d(input_, W_r, [1, 1, 1, 1], padding="SAME")
        conv_U_r = tf.nn.conv2d(state, U_r, [1, 1, 1, 1], padding="SAME")

        r = conv_W_r + conv_U_r
        r = sigmoid(r)

      with vs.variable_scope("Candidate"):
        #candidate gate kernels
        W = vs.get_variable(
              "W", shape=[self._kernel_size, self._kernel_size, self._input_size, self._hidden_size], dtype=dtype)
        U = vs.get_variable(
              "U", shape=[self._kernel_size, self._kernel_size, self._hidden_size, self._hidden_size], dtype=dtype)

        #convolution operations for candidate gate
        conv_W = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding="SAME")
        conv_U = tf.nn.conv2d(r * state, U, [1, 1, 1, 1], padding="SAME")

        c = conv_W + conv_U
        c = tanh(c)

      new_h = u * state + (1 - u) * c
    return new_h, new_h