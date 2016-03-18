from rcn_cell import GRCUCell, StackedGRCUCell
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from utils import load_pkl
from tensorflow.python.ops.seq2seq import embedding_attention_seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, BasicLSTMCell
from keras.preprocessing import sequence

class Video_Caption_Generator():
  def __init__(self, config):
    self._config = config
    self._kernel_size = 3

    self._train_data, _ = self.get_video_data()
    self._nr_training_examples = self._train_data.shape[0]
    self._train_data = self.shuffle_train_data(self._train_data)

    self._word_to_index, self._index_to_word, self._bias_init_vector, self._caption_matrix, self._longest_caption = \
      self.get_caption_dicts(self._train_data)

    self._nr_words = len(self._word_to_index)

    self._W_emb = tf.get_variable(tf.random_uniform([self._nr_words, self._config.dim_hidden], -0.1, 0.1), name='W_emb')

    self._lstm = BasicLSTMCell(self._config.dim_hidden)

    self._encode_image_W = tf.get_variable(tf.random_uniform([self._config.dim_video, self._config.dim_hidden], -0.1,
                                                            0.1), name='encode_image_W')
    self._encode_image_b = tf.get_variable(tf.zeros([self._config.dim_hidden]), name='encode_image_b')

    self._embed_word_W = tf.get_variable(tf.random_uniform([self._config.dim_hidden, self._nr_words], -0.1, 0.1),
                                    name='embed_word_W')
    self._embed_word_b = tf.get_variable(tf.zeros([self._nr_words]), name='embed_word_b')

    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * self._config.learning_rate_decay_factor)

  def shuffle_train_data(self, train_data):
    index = list(train_data.index)
    np.random.shuffle(index)
    train_data = train_data.ix[index]

    return train_data


  def get_video_data(self, train_ratio=0.9):
    video_data = pd.read_csv(self._config.video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.pkl', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(self._config.feats_dir, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data


  def create_vocab(self, captions, word_count_threshold=5): # borrowed this function from NeuralTalk):
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nr_captions = 0
    longest_caption = 0
    for caption in captions:
        nr_captions += 1
        words = caption.lower().split(' ')

        if (longest_caption < len(words)):
          longest_caption = len(words)

        for word in words:
           word_counts[word] = word_counts.get(word, 0) + 1

    vocab = [word for word in word_counts if word_counts[word] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    index_to_word = {}
    index_to_word[0] = '.'  # period at the end of the sentence. make first dimension be end token
    word_to_index = {}
    word_to_index['#START#'] = 0 # make first vector be the start token
    index = 1
    for word in vocab:
        word_to_index[word] = index
        index_to_word[index] = word
        index += 1

    word_counts['.'] = nr_captions
    bias_init_vector = np.array([1.0 * word_counts[index_to_word[index]] for index in index_to_word])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return word_to_index, index_to_word, bias_init_vector


  def get_caption_dicts(self, train_data):
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)

    word_to_index, index_to_word, bias_init_vector, longest_caption = self.create_vocab(captions, word_count_threshold=10)

    captions_ind = map(lambda cap: [word_to_index[word] for word in cap.lower().split(' ')[:-1] if word in word_to_index], captions)

    caption_matrix = sequence.pad_sequences(captions_ind, padding='post', maxlen=longest_caption)

    if not os.path.exists(self._config.index_to_word_path):
      np.save(self._config.index_to_word_path, index_to_word)

    self._config.nr_words = len(word_to_index)

    return word_to_index, index_to_word, bias_init_vector, caption_matrix, longest_caption

  def get_example(self):

    current_train_data = self._train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    current_train_data = current_train_data.reset_index(drop=True)

    current_example_index = np.random.randint(0, self._nr_training_examples)
    current_example = current_train_data[current_example_index]

    vid = current_example['video_path']
    current_caption = self._caption_matrix[current_example_index]

    current_feats_segments = load_pkl(vid)
    # feat_maps_batch_segments = zip(*current_feats_vals)
    #
    # feat_maps_batch_segments = map(lambda segment: zip(*segment), feat_maps_batch_segments)
    # feat_maps_batch_segments = map(lambda segment: map(lambda feat_map: np.asarray(feat_map), segment),
    #                                feat_maps_batch_segments)

    return current_feats_segments, current_caption.tolist()

  def inference_loss(self, encoder_inputs, feed_previous=False):

    decoder_inputs_placeholders = self._longest_caption * tf.placeholder(tf.int32, [self._config.dim_hidden])
    encoder_inputs = np.asarray(encoder_inputs) # list - train_segments long - of inputs of size dim_video
    encoder_inputs = tf.convert_to_tensor(encoder_inputs) # shape = [train_segments, dim_video]

    image_emb = tf.nn.xw_plus_b(encoder_inputs, self._encode_image_W, self._encode_image_b)

    state = tf.zeros([self._lstm.state_size])

    # loss = 0.0
    probs = []

    for time_step in range(self._config.train_segments): ## Phase 1 => only read videos
      if time_step > 0:
        tf.get_variable_scope().reuse_variables()

        output_image, state = self._lstm(image_emb[time_step,:], state)

    tf.get_variable_scope().reuse_variables()
    current_embed = tf.nn.embedding_lookup(self._W_emb, 0) # start token
    output, state = self._lstm(current_embed, state)


    for time_step in range(self._longest_caption + 1): ## Phase 2 => only generate captions
      tf.get_variable_scope().reuse_variables()

      if time_step == 0:
        current_embed = tf.nn.embedding_lookup(self._W_emb, 0) # start token
        # label = 0;
      else:
        current_embed = tf.nn.embedding_lookup(self._W_emb, decoder_inputs_placeholders[time_step])
        # label = decoder_inputs_placeholders[time_step]


      output, state = self._lstm(current_embed, state)
      logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
      probs.append(logit_words)

      # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, label)
      # current_loss = tf.reduce_sum(cross_entropy)
      # loss += current_loss

    return probs, decoder_inputs_placeholders


  def loss(self, logits, labels_placeholder):

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels_placeholder, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss


  def train(self, total_loss, global_step):

    tf.scalar_summary('learning_rate', self._config.learning_rate)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = self.add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
      # opt = tf.train.GradientDescentOptimizer(self._config.learning_rate)
      opt = tf.train.AdamOptimizer(self._config.learning_rate)
      # opt = tf.train.AdamOptimizer(conv_config.INITIAL_LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)
      grads_and_vars = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradients_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    self.add_histograms(grads_and_vars)

    variables_averages_op = self.add_moving_averages_to_all(global_step)

    with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op


  def get_learning_rate_decay_op(self):
    return self.learning_rate_decay_op

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
    var = self.variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
      tf.add_to_collection('losses', weight_decay)
    return var


  def variable_on_cpu(self, name, shape, initializer):
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