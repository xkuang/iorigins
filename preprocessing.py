import cv2
import os
import ipdb
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('nr_frames', 80,
                           """Nr of sample frames at equally-space intervals.""")

tf.app.flags.DEFINE_string('videos_dir', '/Users/ioanaveronicachelu/PycharmProjects/iorigins/media',
                           """youtube clips path""")

tf.app.flags.DEFINE_string('feats_dir', './feats',
                           """youtube features path""")

def main():
  videos = os.listdir(FLAGS.videos_dir)
  videos = filter(lambda x: x.endswith('avi'), videos)

  for video in videos:
    print video

    if os.path.exists (os.path.join(FLAGS.feats_dir, video)):
      print "Already processed ... "
      continue

    video_fullpath = os.path.join(FLAGS.videos_dir, video)

    try:
      cap = cv2.VideoCapture (video_fullpath)
    except:
      ipdb.set_trace()

    frame_count = 0
    frame_list = []

    while True:
      # Capture frame-by-frame
      ret, frame = cap.read()

      if ret is False:
          break
      else:
        print "ret is True"

      frame_list.append(frame)
      frame_count += 1

    frame_list = np.array(frame_list)

    if frame_count > 80:
      frame_indices = np.linspace(0, frame_count, num=FLAGS.nr_frames, endpoint=False).astype(int)
      frame_list = frame_list[frame_indices]

    # cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
    # feats = cnn.get_features(cropped_frame_list)



if __name__=="__main__":
    main()