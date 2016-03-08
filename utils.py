import numpy as np
import cPickle

def create_batches(dataset_size, batch_size):
  """Create batche. Outputs a list [b1, b2...bn], where bi is a list of indices.

  Args:
    dataset_size: the size of the whole data.
    batch_size: the size of each batch.

  Returns:
    output [m1, m2, m3, ..., mk] where mk is a list of indices
  """
  assert dataset_size >= batch_size
  nr_batches = dataset_size / batch_size
  leftover = dataset_size % batch_size
  indices = range(dataset_size)
  if leftover == 0:
      batch_indices = np.split(np.asarray(indices), nr_batches)
  else:
      print 'uneven minibath chunking, overall %d, last one %d' % (batch_size, leftover)
      batch_indices = np.split(np.asarray(indices)[:-leftover], nr_batches)
      batch_indices = batch_indices + [np.asarray(indices[-leftover:])]
  batch_indices = [ind.tolist() for ind in batch_indices]
  return batch_indices

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()