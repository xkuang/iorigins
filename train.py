import os
import pandas as pd
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('video_data_path', './data/video_corpus.csv',
                           """path to video corpus""")

tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/media',
                           """youtube clips path""")

tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats',
                           """youtube features path""")


def get_video_data(train_ratio=0.9):
    video_data = pd.read_csv(FLAGS.video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.pkl', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(FLAGS.feats_dir, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data


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


def main(_):
  train_data, _ = get_video_data()
  captions = train_data['Description'].values
  captions = map(lambda x: x.replace('.', ''), captions)
  captions = map(lambda x: x.replace(',', ''), captions)
  word_to_index, index_to_word, bias_init_vector = create_vocab(captions, word_count_threshold=10)


if __name__ == '__main__':
  tf.app.run()