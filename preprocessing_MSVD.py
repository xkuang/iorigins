import cv2
import os
import ipdb
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from download_cnn import Inception
from vgg import VGG
from utils import dump_pkl

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('nr_frames', 80,
                           """Nr of sample frames at equally-space intervals.""")

tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/media',
                           """youtube clips path""")
# tf.app.flags.DEFINE_string('videos_dir', '/Volumes/Elements/media',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats_vgg',
                           """youtube features path""")
# tf.app.flags.DEFINE_string('feats_dir', '/Volumes/Elements/feats_vgg',
#                            """youtube features path""")
#switch to inception for inception
tf.app.flags.DEFINE_string('cnn_type', 'vgg',
                           """the cnn to get the feature_maps_from""")
tf.app.flags.DEFINE_string('image_size', 224,
                           """the size of the image that goes into the VGG net""")
tf.app.flags.DEFINE_string('nr_feat_maps', 5,
                           """the number of feature maps to get from the cnn""")
tf.app.flags.DEFINE_string('tensor_names', ["import/pool2:0", "import/pool3:0", "import/pool4:0", "import/pool5:0", "import/Relu_1:0"],
                           """the names of the tensors to run in the vgg network""")


def main():
  if FLAGS.cnn_type == 'inception':
    cnn = Inception()
  else:
    cnn = VGG(FLAGS.nr_feat_maps, FLAGS.tensor_names, FLAGS.image_size)
    # cnn.printTensors()

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

      frame_list.append(frame)
      frame_count += 1

    frame_list = np.array(frame_list)

    if frame_count > 80:
      frame_indices = np.linspace(0, frame_count, num=FLAGS.nr_frames, endpoint=False).astype(int)
      frame_list = frame_list[frame_indices]

    cropped_frame_list = np.array(map(lambda x: cnn.preprocess_frame(x), frame_list))
    feats = cnn.get_features(cropped_frame_list)
    # feats_stacked = np.row_stack(feats)
    save_full_path = os.path.join(FLAGS.feats_dir, video + '.pkl')
    dump_pkl(feats, save_full_path)
    # np.save(save_full_path, feats_stacked)

if __name__=="__main__":
    main()