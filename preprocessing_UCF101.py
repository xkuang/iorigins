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

tf.app.flags.DEFINE_string('nr_frames', 10,
                           """Nr of sample frames cropped.""")

# tf.app.flags.DEFINE_string('videos_dir', '/media/ioana/7ED0-6463/UCF-101',
#                            """youtube clips path""")
tf.app.flags.DEFINE_string('videos_dir', './UCF-101',
                           """youtube clips path""")

# tf.app.flags.DEFINE_string('feats_dir', '/media/ioana/7ED0-6463/feats_ucf',
#                            """youtube features path""")
tf.app.flags.DEFINE_string('feats_dir', './feats_ucf',
                           """youtube features path""")

#switch to inception for inception
tf.app.flags.DEFINE_string('cnn_type', 'vgg',
                           """the cnn to get the feature_maps_from""")
tf.app.flags.DEFINE_string('image_size', 224,
                           """the size of the image that goes into the VGG net""")
tf.app.flags.DEFINE_string('nr_feat_maps', 5,
                           """the number of feature maps to get from the cnn""")
tf.app.flags.DEFINE_string('tensor_names', ["import/pool2:0", "import/pool3:0", "import/pool4:0", "import/pool5:0", "import/Relu_1:0"],
                           """the names of the tensors to run in the vgg network""")
tf.app.flags.DEFINE_string('train_test_split', './ucfTrainTestlist/trainlist01.txt',
                           """file where the train-test split info resides""")
tf.app.flags.DEFINE_string('train_data_file', './ucfTrainTestlist/train_data.csv',
                           """file where the train-test split info resides""")
tf.app.flags.DEFINE_string('cropping_sizes', [240, 224, 192, 168],
                           """the cropping sizes to randomly sample from""")


def process_record(cnn, data):
  if os.path.exists (os.path.join(FLAGS.feats_dir, data['feat_path'])):
    print "Already processed ... "
    return

  video_fullpath = os.path.join(FLAGS.videos_dir, data['video_path'])

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
    return

  frame_list = np.array(frame_list)

  if frame_count > FLAGS.nr_frames:
    start = np.random.randint(0, frame_count - 10)
    frame_indices = np.arange(start, start + 10)
    frame_list = frame_list[frame_indices]

  cropped_frame_list = np.array(map(lambda x: cnn.preprocess_frame(FLAGS.cropping_sizes, x), frame_list))
  feats = cnn.get_features(cropped_frame_list)
  # feats_stacked = np.row_stack(feats)
  save_full_path = os.path.join(FLAGS.feats_dir, data['feat_path'])
  dump_pkl(feats, save_full_path)
  # np.save(save_full_path, feats_stacked)

def main():
  if FLAGS.cnn_type == 'inception':
    cnn = Inception()
  else:
    cnn = VGG(FLAGS.nr_feat_maps, FLAGS.tensor_names, FLAGS.image_size)
    # cnn.printTensors()

  dict = {}
  dict["video_path"] = []
  dict["label"] = []
  dict["feat_path"] = []
  with open(FLAGS.train_test_split) as f:
    for line in f:
      video_path, label = line.split()
      dict["video_path"].append(video_path)
      dict["label"].append(label)
      dict["feat_path"].append(video_path.split("/")[-1] + ".pkl")

  train_data = pd.DataFrame(data=dict, columns=['video_path', 'label', "feat_path"])

  if not os.path.exists (FLAGS.train_data_file):
    train_data.to_csv(FLAGS.train_data_file, sep=',')

  train_data.apply(lambda row: process_record(cnn, row), axis=1)


if __name__=="__main__":
    main()