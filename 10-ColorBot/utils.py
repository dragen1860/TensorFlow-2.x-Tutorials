import  os, six, time
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import  urllib


def parse(line):
    """
    Parse a line from the colors dataset.
    """

    # Each line of the dataset is comma-separated and formatted as
    #    color_name, r, g, b
    # so `items` is a list [color_name, r, g, b].
    items = tf.string_split([line], ",").values
    rgb = tf.strings.to_number(items[1:], out_type=tf.float32) / 255.
    # Represent the color name as a one-hot encoded character sequence.
    color_name = items[0]
    chars = tf.one_hot(tf.io.decode_raw(color_name, tf.uint8), depth=256)
    # The sequence length is needed by our RNN.
    length = tf.cast(tf.shape(chars)[0], dtype=tf.int64)
    return rgb, chars, length


def maybe_download(filename, work_directory, source_url):
    """
    Download the data from source url, unless it's already here.
    Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
    Returns:
      Path to resulting file.
    """
    if not tf.io.gfile.exists(work_directory):
        tf.io.gfile.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not tf.io.gfile.exists(filepath):
        temp_file_name, _ = urllib.request.urlretrieve(source_url)
        tf.io.gfile.copy(temp_file_name, filepath)
        with tf.io.gfile.GFile(filepath) as f:
            size = f.size()
            print("Successfully downloaded", filename, size, "bytes.")
    return filepath


def load_dataset(data_dir, url, batch_size):
    """Loads the colors data at path into a PaddedDataset."""

    # Downloads data at url into data_dir/basename(url). The dataset has a header
    # row (color_name, r, g, b) followed by comma-separated lines.
    path = maybe_download(os.path.basename(url), data_dir, url)

    # This chain of commands loads our data by:
    #   1. skipping the header; (.skip(1))
    #   2. parsing the subsequent lines; (.map(parse))
    #   3. shuffling the data; (.shuffle(...))
    #   3. grouping the data into padded batches (.padded_batch(...)).
    dataset = tf.data.TextLineDataset(path).skip(1).map(parse).shuffle(
                buffer_size=10000).padded_batch(
                batch_size, padded_shapes=([None], [None, None], []))
    return dataset