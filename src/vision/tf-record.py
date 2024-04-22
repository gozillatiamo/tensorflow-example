import tensorflow as tf
import tensorflow_datasets as tfds


def print_info():
    data, info = tfds.load("mnist", with_info=True)
    print(info)

def read_tfrecord():
    filename = "/home/gozillatiamo/tensorflow_datasets/mnist/3.0.1/mnist-test.tfrecord-00000-of-00001"
    feature_description = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64)
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(_parse_function)
    for parsed_record in parsed_dataset.take(1):
        print(parsed_record)

# print_info()
read_tfrecord()
