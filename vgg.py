import tensorflow as tf
import cv2
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('vgg_model_dir', "./model/vgg16.tfmodel",
                           """Path to the model needed to create the graph.""")

class VGG(object):
  def __init__(self, nr_feat_maps, tensor_names, image_size):
    self.nr_feat_maps = nr_feat_maps
    self.tensor_names = tensor_names
    self.image_size = image_size
    self.create_graph()

  def create_graph(self):
    with open(FLAGS.vgg_model_dir, mode='rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      images = tf.placeholder("float", [None, self.image_size, self.image_size, 3])
      _ = tf.import_graph_def(graph_def, input_map={ "images": images })
      print "graph loaded from disk"

  def preprocess_frame(self, image):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    # image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
      resized_image = cv2.resize(image, (self.image_size, self.image_size))

    elif height < width:
      resized_image = cv2.resize(image, (int(width * float(self.image_size)/height), self.image_size))
      cropping_length = int((resized_image.shape[1] - self.image_size) / 2)
      resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
      resized_image = cv2.resize(image, (self.image_size, int(height * float(self.image_size) / width)))
      cropping_length = int((resized_image.shape[0] - self.image_size) / 2)
      resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (self.image_size, self.image_size))

  def printTensors(self):
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
      init = tf.initialize_all_variables()
      sess.run(init)
      print "variables initialized"

      for frame in frame_list:
        image_data = tf.convert_to_tensor(frame, dtype=tf.float32).eval()
        image_data = np.reshape(image_data, [1, self.image_size, self.image_size, 3])
        assert image_data.shape == (1, self.image_size, self.image_size, 3)
        # DT_FLOAT
        images = tf.placeholder("float", [None, self.image_size, self.image_size, 3])

        feed_dict = { images: image_data }

        tensor_list = []

        for tensor in self.tensor_names:
          tensor_list.append(sess.graph.get_tensor_by_name(tensor))

        feat_map_list = sess.run(tensor_list, feed_dict=feed_dict)

        video_feat_maps.append(feat_map_list)

    return video_feat_maps
