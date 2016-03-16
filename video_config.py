class ActionConfig(object):
  #Directory where to write event logs
  train_dir = './train_UCF101'

  #Directory where to write event logs for evaluation step
  eval_dir = './eval_UCF101'

  #path to video corpus
  video_data_path = './ucfTrainTestlist/train_data.csv'

  #path to video corpus
  test_data_path = './ucfTrainTestlist/test_data.csv'

  #youtube clips path
  # videos_dir = './UCF-101'
  videos_dir = '/media/ioana/7ED0-6463/UCF-101'

  #youtube features path
  # feats_dir = './feats_ucf'
  feats_dir = '/media/ioana/7ED0-6463/feats_ucf'

  #the size of the input image/frame
  input_sizes = [[56, 56, 128],
                 [28, 28, 256],
                 [14, 14, 512],
                 [7,  7, 512],
                 [1, 1, 4096]]

  #youtube features path
  hidden_sizes = [64, 128, 256, 256, 512]

  #Nr of batches
  batch_size_train = 8

  #Nr of batches
  batch_size_test = 8

  #Nr of sample frames at equally-space intervals
  nr_frames = 10

  #Nr of classes
  nr_classes = 101

  #Nr of feature maps extracted from the inception CNN for each frame
  nr_feat_maps = 5

  #Nr of epochs to train
  max_steps = 1000

  #Model's learning rate
  learning_rate = 0.001

  #Model's learning rate decay factor
  learning_rate_decay_factor = 0.6

  #Dropout ration for the last layer of the classifiers
  keep_prob = 0.7

  #Number of epochs per decay of the learning rate
  nr_epochs_per_decay = 350

  #Moving average decay rate
  moving_average_decay = 0.9999

  #Whether to log device placement
  log_device_placement = False

  #Variable to specify if the last model should be resumed or a new one created
  resume = False

  #the size of the image that goes into the VGG net
  image_size = 224

  #the names of the tensors to run in the vgg network
  tensor_names = ["import/pool2:0", "import/pool3:0", "import/pool4:0", "import/pool5:0", "import/Relu_1:0"]

  #The number of segments to extract 10 frames to test from
  test_segments = 5

  #the cropping sizes to randomly sample from
  cropping_sizes = [240, 224, 192, 168]

  #Number of examples to run for eval
  nr_test_examples = 10000

  #Whether to run eval only once
  run_once = False

  #How often to run the eval
  eval_interval_secs = 60 * 5

  #Use Stacked GRCU instead of regular ones
  stacked = True

class CaptionConfig(object):
  #Directory where to write event logs
  train_dir = './train_MSVD'

  #Directory where to write event logs for evaluation step
  eval_dir = './eval_MSVD'

  #path to video corpus
  video_data_path = './data/video_corpus.csv'

  #path to video corpus
  # test_data_path = './ucfTrainTestlist/test_data.csv'

  #youtube clips path
  # videos_dir = './UCF-101'
  videos_dir = '/media/ioana/7ED0-6463/MSVD'

  #youtube features path
  # feats_dir = './feats_ucf'
  feats_dir = '/media/ioana/7ED0-6463/feats_msvd'

  #index_to_word dictionary path
  index_to_word_path = '/media/ioana/7ED0-6463/index_to_word_MSVD/dict.npy'

  #the size of the input image/frame
  input_sizes = [[56, 56, 128],
                 [28, 28, 256],
                 [14, 14, 512],
                 [7,  7, 512],
                 [1, 1, 4096]]

  #youtube features path
  hidden_sizes = [64, 128, 256, 256, 512]

  #Nr of batches
  batch_size_train = 8

  #Nr of batches
  batch_size_test = 8

  #Nr of sample frames at equally-space intervals
  nr_frames = 10

  #Nr of classes
  nr_classes = 101

  #Nr of feature maps extracted from the inception CNN for each frame
  nr_feat_maps = 5

  #Nr of epochs to train
  max_steps = 1000

  #Model's learning rate
  learning_rate = 0.001

  #Model's learning rate decay factor
  learning_rate_decay_factor = 0.6

  #Dropout ration for the last layer of the classifiers
  keep_prob = 0.7

  #Number of epochs per decay of the learning rate
  nr_epochs_per_decay = 350

  #Moving average decay rate
  moving_average_decay = 0.9999

  #Whether to log device placement
  log_device_placement = False

  #Variable to specify if the last model should be resumed or a new one created
  resume = False

  #the size of the image that goes into the VGG net
  image_size = 224

  #the names of the tensors to run in the vgg network
  tensor_names = ["import/pool2:0", "import/pool3:0", "import/pool4:0", "import/pool5:0", "import/Relu_1:0"]

  #The number of segments to extract 10 frames to test from
  test_segments = 5

  #Nr of segments to sample frames from
  train_segments = 5

  #the cropping sizes to randomly sample from
  cropping_sizes = [240, 224, 192, 168]

  #Number of examples to run for eval
  nr_test_examples = 10000

  #Whether to run eval only once
  run_once = False

  #How often to run the eval
  eval_interval_secs = 60 * 5

  #Use Stacked GRCU instead of regular ones
  stacked = True

  #The buckets needed to pad the input captions
  buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
