class Spatio_Temporal_Generator():
   def __init__(self, dim_image, nr_words, dim_hidden, batch_size_train, nr_frames, bias_init_vector=None):
        self.dim_image = dim_image
        self.nr_words = nr_words
        self.dim_hidden = dim_hidden
        self.batch_size_train = batch_size_train
        self.nr_frames = nr_frames