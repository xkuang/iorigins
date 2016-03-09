from rcn_cell import GRCUCell

class Spatio_Temporal_Generator():
   def __init__(self, image_size, nr_words, batch_size_train, nr_frames, nr_feat_maps, bias_init_vector=None):
     self.image_size = image_size
     self.nr_words = nr_words
     self.batch_size_train = batch_size_train
     self.nr_frames = nr_frames
     self.nr_feat_maps = nr_feat_maps

     self.gcru = GRCUCell(dim_hidden)
