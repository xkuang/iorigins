from rcn_cell import GRCUCell
import tensorflow as tf

class Action_Recognizer():
  def __init__(self, input_sizes, hidden_sizes, batch_size_train, nr_frames, nr_feat_maps, nr_classes):
    self.input_sizes = input_sizes
    self.hidden_sizes = hidden_sizes
    self.batch_size_train = batch_size_train
    self.nr_frames = nr_frames
    self.nr_feat_maps = nr_feat_maps
    self.kernel_size = 3
    self.nr_classes = nr_classes

    self.grcu_list = []

    for i, hidden_layer_size in enumerate(self.hidden_sizes):
      self.gcru_list.append(GRCUCell(hidden_layer_size,
                                     self.input_sizes[i][0],
                                     self.input_sizes[i][1],
                                     self.input_sizes[i][2],
                                     self.kernel_size))


  def build_model(self):
    #feature map placeholders
    feat_map_placeholders = []
    for input_size in self.input_sizes:
      feat_map_placeholders.append(tf.placeholder(tf.float32, [self.nr_frames,
                                                               self.input_size[0],
                                                               self.input_size[1],
                                                               self.input_size[2]]))

    internal_states = []
    for grcu in self.grcu_list:
      state_size = [grcu.state_size[0], grcu.state_size[1], grcu.state_size[2]]
      internal_states.append(tf.zeros(state_size))

    outputs = []
    for i in range(self.nr_frames):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
        for i, grcu in enumerate(self.grcu_list):
          with tf.variable_scope("GRU-RCN%d" % (i)):
            output, internal_states = self.grcu( feat_map_placeholders[i,:,:,:], internal_states[i] )
            outputs.append(output)

    output_embeds = []
    for i, internal_state in internal_states:
      avg_pool = tf.nn.avg_pool(internal_state,
                                ksize=[1, self.input_sizes[i][0], self.input_sizes[i][1], 1],
                                strides=[1, 1, 1, 1], padding='SAME', name=('avg_pool' + i))
      dropout = tf.nn.dropout(avg_pool, 0.5)

      with tf.variable_scope("softmax_linear" + i) as scope:
        weights_soft = self.variable_with_weight_decay("weights", [ self.input_sizes[i][2], self.nr_classes],
                                          stddev=0.07, wd=0.004)
        biases_soft = self.variable_on_cpu("biases", [self.nr_classes],
                                  tf.constant_initializer(0.1))
        softmax_linear = tf.nn.xw_plus_b(dropout, weights_soft, biases_soft, name=scope.name)
        output_embeds.append(softmax_linear)
        # conv_summary.activation_summary(softmax_linear)


  def variable_with_weight_decay(name, shape, stddev, wd):
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
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
      tf.add_to_collection('losses', weight_decay)
    return var


  def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
    return var