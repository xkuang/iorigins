import cv2
import os
import ipdb
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from inception import Inception
from vgg import VGG
from utils import dump_pkl

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('nr_frames', 10,
                           """Nr of sample frames at equally-space segments.""")
tf.app.flags.DEFINE_string('nr_segments', 5,
                           """Nr of segments to sample frames from.""")
tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/Elements/MSVD',
                           """youtube clips path""")
# tf.app.flags.DEFINE_string('videos_dir', '/Volumes/Elements/media',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/Elements/feats_MSVD',
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
tf.app.flags.DEFINE_string('cropping_sizes', [240, 224, 192, 168],
                           """the cropping sizes to randomly sample from""")


def main():
  # inception = Inception()
  vgg = VGG(FLAGS.nr_feat_maps, FLAGS.tensor_names, FLAGS.image_size)

  videos = os.listdir(FLAGS.videos_dir)
  videos = filter(lambda x: x.endswith('avi'), videos)

  for video in videos:
    # print video

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

    if frame_count == 0:
      print "This video could not be processed"
      continue

    frame_list = np.array(frame_list)

    if frame_count < FLAGS.nr_segments * FLAGS.nr_frames:
      print ("This video is too short. It has %d frames" % frame_count)

    segment_indices = np.linspace(0, frame_count, num=FLAGS.nr_segments, endpoint=False).astype(int)

    segment_list = []
    for segment_idx in segment_indices:
      segment_frames = frame_list[segment_idx : (segment_idx + FLAGS.nr_frames)]
      cropped_segment_frames = np.array(map(lambda x: vgg.preprocess_frame(FLAGS.cropping_sizes, x), segment_frames))
      segment_feats = vgg.get_features(cropped_segment_frames)
      shape = segment_feats[4].shape
      segment_feats[4] = np.reshape(segment_feats[4], [shape[0], shape[1], 1, 1, shape[2]])
      segment_list.append(segment_feats)

    save_full_path = os.path.join(FLAGS.feats_dir, video + '.pkl')
    dump_pkl(segment_list, save_full_path)

if __name__=="__main__":
    main()