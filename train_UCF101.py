import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl
import random
import time
import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_UCF101',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('video_data_path', './ucfTrainTestlist/train_data.csv',
                           """path to video corpus""")

# tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/UCF101',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/7ED0-6463/UCF-101',
                           """youtube clips path""")

# tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats_ucf',
#                            """youtube features path""")
tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/7ED0-6463/feats_ucf',
                           """youtube features path""")

tf.app.flags.DEFINE_string('index_to_word_dir', '/media/ioana/Elements/index_to_word',
                           """index_to_word dictionary path""")

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

def get_video_data():
    video_data = pd.read_csv(FLAGS.video_data_path, sep=',')
    video_data['feat_path'] = video_data['feat_path'].map(lambda x: os.path.join(FLAGS.feats_dir, x))
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(FLAGS.videos_dir, x))
    video_data = video_data[video_data['feat_path'].map(lambda x: os.path.exists( x ))]

    # unique_filenames = video_data['video_path'].unique()

    return video_data


def create_vocab(captions, word_count_threshold=5): # borrowed this function from NeuralTalk
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


def shuffle_train_data(train_data):
  index = list(train_data.index)
  np.random.shuffle(index)
  train_data = train_data.ix[index]

  return train_data


def train():
  train_data = get_video_data()
  nr_training_examples = train_data.shape[0]

  model = Action_Recognizer(
            input_sizes=FLAGS.input_sizes,
            hidden_sizes=FLAGS.hidden_sizes,
            batch_size_train=FLAGS.batch_size_train,
            nr_frames=FLAGS.nr_frames,
            nr_feat_maps=FLAGS.nr_feat_maps,
            nr_classes=FLAGS.nr_classes,
            keep_prob=FLAGS.keep_prob,
            nr_training_examples=nr_training_examples,
            nr_epochs_per_decay=FLAGS.nr_epochs_per_decay,
            moving_average_decay=FLAGS.moving_average_decay,
            initial_learning_rate=FLAGS.learning_rate,
            learning_rate_decay_factor=FLAGS.learning_rate_decay_factor)

  global_step = tf.Variable(0, trainable=False)

  train_data = shuffle_train_data(train_data)

  current_batch_indices = random.sample(xrange(nr_training_examples), 64)
  current_batch = train_data.ix[current_batch_indices]

  current_videos = current_batch['video_path'].values
  current_feats = current_batch['feat_path'].values
  labels = current_batch['label'].values

  current_feats_vals = map(lambda vid: load_pkl(vid), current_feats)
  feat_maps_batch =  zip(*current_feats_vals)

  feat_maps_batch = map(lambda x: np.asarray(x), feat_maps_batch)
  shape = feat_maps_batch[4].shape
  feat_maps_batch[4] = np.reshape(feat_maps_batch[4], [shape[0], shape[1], 1, 1, shape[2]])

  logits = model.inference()

  loss = model.loss(logits, labels)

  train_op = model.train(loss, global_step)

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables())

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Start running operations on the Graph.
  sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
  sess.run(init)

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  graph_def = sess.graph.as_graph_def(add_shapes=True)
  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                          graph_def=graph_def)

  for step in xrange(FLAGS.max_steps):
    print ("epoch %d" % step)

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