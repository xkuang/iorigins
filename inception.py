from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
# import cv
import cv2
# import ipdb
import skimage
import pylab as plt

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', './model',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

class Inception(object):
  def __init__(self):
    self.create_graph()

  def create_graph(self):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')

  def preprocess_frame(self, image, target_height=299, target_width=299):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    # image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
      resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
      resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
      cropping_length = int((resized_image.shape[1] - target_height) / 2)
      resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
      resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
      cropping_length = int((resized_image.shape[0] - target_width) / 2)
      resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

  def printInceptionTensors(self):
    graph = tf.get_default_graph()
    operations = graph.get_operations()
    for operation in operations:
      print ("Operation:", operation.name)
      for k in operation.inputs:
          print (operation.name, "Input ", k.name, k.get_shape())
      for k in operation.outputs:
          print (operation.name, "Output ", k.name)
      print ("\n")

  def get_features(self, frame_list):
    video_feat_maps = []
    with tf.Session() as sess:
      # input_tensor = sess.graph.get_tensor_by_name('Mul:0')
      # print(input_tensor.get_shape())

      for frame in frame_list:
        image_data = tf.convert_to_tensor(frame, dtype=tf.float32).eval()
        # cv2.imshow('image', first_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image_data = np.reshape(image_data, [1, 299, 299, 3])

        # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        #sess.graph.get_operations()

        tensor_list = []

        tensor_list.append(sess.graph.get_tensor_by_name('pool:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('pool_1:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_1/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_2/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_3/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_4/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_5/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_6/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_7/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_8/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('mixed_9/join:0'))
        tensor_list.append(sess.graph.get_tensor_by_name('pool_3:0'))

        feat_map_list = sess.run(tensor_list,
                               {'Mul:0': image_data})

        video_feat_maps.append(feat_map_list)
        # print ("pool0 : ", feat_map_list[0].shape)
        # print ("pool1 : ", feat_map_list[1].shape)
        # print ("mixed0 : ", feat_map_list[2].shape)
        # print ("mixed1 : ", feat_map_list[3].shape)
        # print ("mixed2 : ", feat_map_list[4].shape)
        # print ("mixed3 : ", feat_map_list[5].shape)
        # print ("mixed4 : ", feat_map_list[6].shape)
        # print ("mixed5 : ", feat_map_list[7].shape)
        # print ("mixed6 : ", feat_map_list[8].shape)
        # print ("mixed7 : ", feat_map_list[9].shape)
        # print ("mixed8 : ", feat_map_list[10].shape)
        # print ("mixed9 : ", feat_map_list[11].shape)
        # print ("pool3 : ", feat_map_list[12].shape)
    return video_feat_maps

#
# def run_inference_on_image(image):
#   """Runs inference on an image.
#
#   Args:
#     image: Image file name.
#
#   Returns:
#     Nothing
#   """
#   if not tf.gfile.Exists(image):
#     tf.logging.fatal('File does not exist %s', image)
#   image_data = tf.gfile.FastGFile(image, 'rb').read()
#
#   # Creates graph from saved GraphDef.
#   create_graph()
#
#   with tf.Session() as sess:
#     # Some useful tensors:
#     # 'softmax:0': A tensor containing the normalized prediction across
#     #   1000 labels.
#     # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
#     #   float description of the image.
#     # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
#     #   encoding of the image.
#     # Runs the softmax tensor by feeding the image_data as input to the graph.
#     softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
#     predictions = sess.run(softmax_tensor,
#                            {'DecodeJpeg/contents:0': image_data})
#     predictions = np.squeeze(predictions)
#
#     # Creates node ID --> English string lookup.
#     node_lookup = NodeLookup()
#
#     top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
#     for node_id in top_k:
#       human_string = node_lookup.id_to_string(node_id)
#       score = predictions[node_id]
#       print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def preprocess_video():
  video_path = "/media/ioana/Elements/media/mv89psg6zh4_33_46.avi"

  cap = cv2.VideoCapture (video_path)

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

  cnn = Inception()

  if frame_count > 80:
      frame_indices = np.linspace(0, frame_count, num=FLAGS.nr_frames, endpoint=False).astype(int)
      frame_list = frame_list[frame_indices]

  cropped_frame_list = np.array(map(lambda x: cnn.preprocess_frame(x), frame_list))

  cnn.create_graph()


  # predictions = sess.run(softmax_tensor,
  #                       {'Mul:0': image_data})
  #
  # predictions = np.squeeze(predictions)
  #
  # # Creates node ID --> English string lookup.
  # node_lookup = NodeLookup()
  #
  # top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
  # for node_id in top_k:
  #   human_string = node_lookup.id_to_string(node_id)
  #   score = predictions[node_id]
  #   print('%s (score = %.5f)' % (human_string, score))




def main(_):
  maybe_download_and_extract()
  # printInceptionTensors()
  preprocess_video()

  # image = (FLAGS.image_file if FLAGS.image_file else
  #          os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  # run_inference_on_image(image)


if __name__ == '__main__':
  tf.app.run()