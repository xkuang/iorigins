from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from test.video2text import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('batch_size_train', 64,
                           """Nr of batches""")
tf.app.flags.DEFINE_string('batch_size_valid', 128,
                           """Nr of batches""")
tf.app.flags.DEFINE_string('batch_size_test', 128,
                           """Nr of batches""")

def prepare_data(engine, video_caption_ids):
  for index, video_caption_id in enumerate(video_caption_ids):
    print ('processed %d/%d pairs of video/caption' % (index, len(video_caption_ids)))
    video_id, caption_id = video_caption_id.split('_')
    print ('here should start processing of video_id %s and caption_id %s' % (video_id, caption_id))


def main(_):
  engine = Video2text (FLAGS.batch_size_train, FLAGS.batch_size_valid, FLAGS.batch_size_test)

  seen_batches = 0
  t = time.time()
  for indices in engine.train_batches:
    seen_batches += 1
    t0 = time.time()
    video_caption_ids = [engine.train[index] for index in indices]

    prepare_data(engine, video_caption_ids)

    print ('seen %d batches, used time %.2f '% (seen_batches, time.time() - t0))

  print ('used time %.2f' % (time.time() - t))


if __name__ == '__main__':
  tf.app.run()