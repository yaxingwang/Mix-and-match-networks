import tensorflow as tf
import random
import os
import pdb

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
    

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_0', '/home/yaxing/CN_dataset/train/blue',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_1', '/home/yaxing/CN_dataset/train/brown',
                       'X input directory, default: data/apple2orange/trainA')

tf.flags.DEFINE_string('input_2', '/home/yaxing/CN_dataset/train/black',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_3', '/home/yaxing/CN_dataset/train/green',
                       'X input directory, default: data/apple2orange/trainA')

tf.flags.DEFINE_string('input_4', '/home/yaxing/CN_dataset/train/grey',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_5', '/home/yaxing/CN_dataset/train/orange',
                       'X input directory, default: data/apple2orange/trainA')

tf.flags.DEFINE_string('input_6', '/home/yaxing/CN_dataset/train/pink',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_7', '/home/yaxing/CN_dataset/train/purple',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_8', '/home/yaxing/CN_dataset/train/red',
                       'X input directory, default: data/apple2orange/trainA')


tf.flags.DEFINE_string('input_9', '/home/yaxing/CN_dataset/train/white',
                       'X input directory, default: data/apple2orange/trainA')

tf.flags.DEFINE_string('input_10', '/home/yaxing/CN_dataset/train/yellow',
                       'X input directory, default: data/apple2orange/trainA')

tf.flags.DEFINE_string('output_0', '../data/tfrecords/domain_0.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_1', '../data/tfrecords/domain_1.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_2', '../data/tfrecords/domain_2.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_3', '../data/tfrecords/domain_3.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_4', '../data/tfrecords/domain_4.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_5', '../data/tfrecords/domain_5.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_6', '../data/tfrecords/domain_6.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_7', '../data/tfrecords/domain_7.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_8', '../data/tfrecords/domain_8.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_9', '../data/tfrecords/domain_9.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('output_10', '../data/tfrecords/domain_10.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')



def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error, e:
    pass

  images_num = len(file_paths)

  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    
    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Convert 0 data to tfrecords...")
  data_writer(FLAGS.input_0, FLAGS.output_0)
  print("Convert 1 data to tfrecords...")
  data_writer(FLAGS.input_1, FLAGS.output_1)
  print("Convert 2 data to tfrecords...")
  data_writer(FLAGS.input_2, FLAGS.output_2)
  print("Convert 3 data to tfrecords...")
  data_writer(FLAGS.input_3, FLAGS.output_3)
  print("Convert 4 data to tfrecords...")
  data_writer(FLAGS.input_4, FLAGS.output_4)
  print("Convert 5 data to tfrecords...")
  data_writer(FLAGS.input_5, FLAGS.output_5)
  print("Convert 6 data to tfrecords...")
  data_writer(FLAGS.input_6, FLAGS.output_6)
  print("Convert 7 data to tfrecords...")
  data_writer(FLAGS.input_7, FLAGS.output_7)
  print("Convert 8 data to tfrecords...")
  data_writer(FLAGS.input_8, FLAGS.output_8)
  print("Convert 9 data to tfrecords...")
  data_writer(FLAGS.input_9, FLAGS.output_9)
  print("Convert 10 data to tfrecords...")
  data_writer(FLAGS.input_10, FLAGS.output_10)
 # print("Convert Y data to tfrecords...")
  #data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)

if __name__ == '__main__':
  tf.app.run()
