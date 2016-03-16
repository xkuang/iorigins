from rcn_cell import GRCUCell, StackedGRCUCell
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from utils import load_pkl
from tensorflow.python.ops.seq2seq import embedding_attention_seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell

class Video_Caption_Generator():
  def __init__(self, config):
    self._config = config
    self._kernel_size = 3

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
    for caption in captions:
        nr_captions += 1
        for word in caption.lower().split(' '):
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

    word_to_index, index_to_word, bias_init_vector = self.create_vocab(captions, word_count_threshold=10)

    if not os.path.exists(self._config.index_to_word_path):
      np.save(self._config.index_to_word_path, index_to_word)

    self._config.nr_words = len(word_to_index)

    return word_to_index, index_to_word, bias_init_vector, captions


  def get_batch(self):
    train_data, _ = self.get_video_data()
    nr_training_examples = train_data.shape[0]
    train_data = self.shuffle_train_data(train_data)

    # word_to_index, index_to_word, bias_init_vector = self.get_caption_dicts(train_data)

    current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    current_train_data = current_train_data.reset_index(drop=True)

    current_batch_indices = random.sample(xrange(nr_training_examples), self._config.batch_size_train)
    current_batch = current_train_data.ix[current_batch_indices]

    current_feats = current_batch['video_path'].values
    captions = current_batch['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)

    current_feats_vals = map(lambda vid: load_pkl(vid), current_feats)
    feat_maps_batch_segments = zip(*current_feats_vals)

    feat_maps_batch_segments = map(lambda segment: zip(*segment), feat_maps_batch_segments)
    feat_maps_batch_segments = map(lambda segment: map(lambda feat_map: np.asarray(feat_map), segment),
                                   feat_maps_batch_segments)

    return feat_maps_batch_segments, np.asarray(captions)

  def get_example(self):
    train_data, _ = self.get_video_data()
    nr_training_examples = train_data.shape[0]
    train_data = self.shuffle_train_data(train_data)

    # word_to_index, index_to_word, bias_init_vector = self.get_caption_dicts(train_data)

    current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    current_train_data = current_train_data.reset_index(drop=True)

    current_example_index = np.random.randint(0, nr_training_examples)
    current_example = current_train_data[current_example_index]

    vid = current_example['video_path']
    caption = current_example['Description'].value
    caption = map(lambda x: x.replace('.', ''), caption)
    caption = map(lambda x: x.replace(',', ''), caption)
    words = caption.lower().split(' ')

    current_feats_vals = load_pkl(vid)
    # feat_maps_batch_segments = zip(*current_feats_vals)
    #
    # feat_maps_batch_segments = map(lambda segment: zip(*segment), feat_maps_batch_segments)
    # feat_maps_batch_segments = map(lambda segment: map(lambda feat_map: np.asarray(feat_map), segment),
    #                                feat_maps_batch_segments)

    return current_feats_vals, np.asarray(caption)

  def inference(self, encoder_inputs, feed_previous=False):
    cell = GRUCell(self._config.cell_state_size) #, self._config.cell_input_size)

    bucket_decoder_inputs_placeholders = []
    bucket_output = []
    for bucket in self._config.buckets:
      decoder_inputs_placeholders = bucket * tf.placeholder(tf.int32, [self._config.embed_size])
      bucket_decoder_inputs_placeholders.append(decoder_inputs_placeholders)

      output, _ = embedding_attention_seq2seq(encoder_inputs, decoder_inputs_placeholders, cell,
                                           self._config.num_encoder_symbols, self._config.num_decoder_symbols,
                                           feed_previous=feed_previous)
      bucket_output.append(output)

    return bucket_output, bucket_decoder_inputs_placeholders


  # def loss(self, logits):



  # def train(self, total_loss, global_step):




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