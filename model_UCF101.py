from rcn_cell import GRCUCell
import tensorflow as tf

class Action_Recognizer():
  def __init__(self, input_sizes, hidden_sizes, batch_size_train, nr_frames, nr_feat_maps):
    self.input_sizes = input_sizes
    self.hidden_sizes = hidden_sizes
    self.batch_size_train = batch_size_train
    self.nr_frames = nr_frames
    self.nr_feat_maps = nr_feat_maps
    self.kernel_size = 3

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
      output_embeds.append(tf.nn.avg_pool(internal_state, ksize=[1,
                                self.input_sizes[i][0],
                                self.input_sizes[i][1], 1],
                                strides=[1, 1, 1, 1], padding='SAME', name=('avg_pool' + i)))




