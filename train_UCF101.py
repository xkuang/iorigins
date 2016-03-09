import os
import pandas as pd
import tensorflow as tf
import numpy as np
from model_UCF101 import Action_Recognizer
from utils import load_pkl

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('video_data_path', './ucfTrainTestlist/train_data.csv',
                           """path to video corpus""")

# tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/UCF101',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('videos_dir', '/home/ioana/Downloads/UCF-101',
                           """youtube clips path""")

# tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats_ucf',
#                            """youtube features path""")
tf.app.flags.DEFINE_string('feats_dir', '/home/ioana/Downloads/feats_ucf',
                           """youtube features path""")

tf.app.flags.DEFINE_string('index_to_word_dir', '/media/ioana/Elements/index_to_word',
                           """index_to_word dictionary path""")

tf.app.flags.DEFINE_string('input_sizes',  [[56, 56, 128],
                                            [28, 28, 256],
                                            [80, 14, 14, 512],
                                            [80,  7,  7, 512]],
                           """the size of the input image/frame""")
tf.app.flags.DEFINE_string('hidden_sizes', [64, 128, 256, 256, 512],
                           """youtube features path""")
tf.app.flags.DEFINE_string('batch_size_train', 1,
                           """Nr of batches""")
tf.app.flags.DEFINE_string('nr_frames', 10,
                           """Nr of sample frames at equally-space intervals.""")
tf.app.flags.DEFINE_string('nr_classes', 101,
                           """Nr of classes.""")
tf.app.flags.DEFINE_string('nr_feat_maps', 5,
                           """Nr of feature maps extracted from the inception CNN for each frame.""")
tf.app.flags.DEFINE_string('nr_epochs', 1,
                           """Nr of epochs to train.""")
tf.app.flags.DEFINE_string('learning_rate', 0.001,
                           """Model's learning rate.""")


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


def train():
  train_data = get_video_data()

  model = Action_Recognizer(
            input_sizes=FLAGS.input_sizes,
            hidden_sizes=FLAGS.hidden_sizes,
            batch_size_train=FLAGS.batch_size_train,
            nr_frames=FLAGS.nr_frames,
            nr_feat_maps=FLAGS.nr_feat_maps,
            nr_classes=FLAGS.nr_classes)

  for epoch in range(FLAGS.nr_epochs):
    print ("epoch %d", epoch)
    index = list(train_data.index)
    np.random.shuffle(index)
    train_data = train_data.ix[index]

    current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    current_train_data = current_train_data.reset_index(drop=True)

    for start, end in zip(
        range(0, len(current_train_data), FLAGS.batch_size_train),
        range(FLAGS.batch_size_train, len(current_train_data), FLAGS.batch_size_train)):

      current_batch = current_train_data[start:end]
      current_videos = current_batch['video_path'].values
      current_feats = current_batch['feat_path'].values

      # current_feats = np.zeros((FLAGS.batch_size_train, FLAGS.nr_frames, dim_image))
      current_feats_vals = map(lambda vid: load_pkl(vid), current_feats)

      video_feats = current_feats_vals[0]
      print ("nr frames for vid id %s : %d" % (current_batch['VideoID'].values[0], len(video_feats)))
      # current_video_masks = np.zeros((batch_size, n_frame_step))

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()