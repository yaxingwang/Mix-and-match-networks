# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pdb
# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def write_image_annotation_path_to_tfrecord(filename_pairs, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    index = 0
    for  k in xrange(len(filename_pairs)/2):
        i = 2*k
        j = 2*k + 1
        for z in xrange(1):
            image_path1,anno_path1,depth_path1 = filename_pairs[i]
            img_ = np.array(Image.open(image_path1))
            img1 = img_[:,:240,:]
            annotation_ = np.array(Image.open(anno_path1))
            annotation1 = annotation_[:,:240]       
            img_raw1 = img1.tostring()
            annotation_raw1 = annotation1.tostring()
	    depth_ = np.array(Image.open(depth_path1))
            depth= depth_[:,:240]        
            depth = depth.astype('float32')
            depth1 = np.log(depth + 1.)
            depth_raw1 = depth1.tostring()



            image_path2,anno_path2,depth_path2 = filename_pairs[j]
            img_ = np.array(Image.open(image_path2))
            img2 = img_[:,:240,:]
            depth_ = np.array(Image.open(depth_path2))
            depth= depth_[:,:240]        
            depth = depth.astype('float32')
            depth2 = np.log(depth + 1.)
            img_raw2 = img2.tostring()
            depth_raw2 = depth2.tostring()

            annotation_ = np.array(Image.open(anno_path2))
            annotation2 = annotation_[:,:240]        
            annotation_raw2 = annotation2.tostring()


            height1 = img1.shape[0]
            width1 = img1.shape[1]
            height2 = img2.shape[0]
            width2 = img2.shape[1]
            assert height1==height2,'the height1 is not same with height2'
            assert width1==width2,'the width1 is not same with width2'
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height1),
                'width': _int64_feature(width1),
                'image_raw1': _bytes_feature(img_raw1),
                'image_raw2': _bytes_feature(img_raw2),
                'mask_raw1': _bytes_feature(annotation_raw1),
                'mask_raw2': _bytes_feature(annotation_raw2),
                'depth_raw1':_bytes_feature(depth_raw1),
                'depth_raw2':_bytes_feature(depth_raw2)}))
            writer.write(example.SerializeToString())
        print('index: %d'%k)
    writer.close()

def test():

    for img_path, annotation_path,depth_path in filename_pairs:

        img_ = np.array(Image.open(img_path))
	img = img_[:,:240,:]
        annotation_ = np.array(Image.open(annotation_path))
	annotation = annotation_[:,:240]        
        depth_ = np.array(Image.open(depth_path))
	depth= depth_[:,:240]        
	depth = depth.astype('float32')
	depth = np.log(depth + 1.)
        index +=1
	print index

       
        # Unomment this one when working with surgical data
        # annotation = annotation[:, :, 0]

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = depth.shape[0]
        width = depth.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()
        depth_raw = depth.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw),
	    'depth_raw':_bytes_feature(depth_raw)}))

        writer.write(example.SerializeToString())

    writer.close()

def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from
    
    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """
    
    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])

        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])

        depth_string = (example.features.feature['depth_raw']
                                      .bytes_list
                                      .value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        annotation = annotation_1d.reshape((height, width))

        depth_1d = np.fromstring(depth_string, dtype=np.uint8)
        depth = depth_1d.reshape((height, width, -1))

        
        image_annotation_pairs.append((img, annotation,depth))
    
    return image_annotation_pairs


def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw1': tf.FixedLenFeature([], tf.string),
        'image_raw2': tf.FixedLenFeature([], tf.string),
        'mask_raw1': tf.FixedLenFeature([], tf.string),
        'mask_raw2': tf.FixedLenFeature([], tf.string),
        'depth_raw1': tf.FixedLenFeature([], tf.string),
        'depth_raw2': tf.FixedLenFeature([], tf.string)
        })

    
    image1 = tf.decode_raw(features['image_raw1'], tf.uint8)
    image2 = tf.decode_raw(features['image_raw2'], tf.uint8)
    annotation1 = tf.decode_raw(features['mask_raw1'], tf.uint8)
    annotation2 = tf.decode_raw(features['mask_raw2'], tf.uint8)
    depth1 = tf.decode_raw(features['depth_raw1'], tf.float32)
    depth2 = tf.decode_raw(features['depth_raw2'], tf.float32)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    
    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension

    annotation_shape = tf.stack([height, width, 1])
    depth_shape = tf.stack([height, width, 1])
    #pdb.set_trace()
    image1 = tf.reshape(image1, image_shape)
    image2 = tf.reshape(image2, image_shape)
    annotation1 = tf.reshape(annotation1, annotation_shape)
    annotation2 = tf.reshape(annotation2, annotation_shape)
    depth1 = tf.reshape(depth1, depth_shape)
    depth2 = tf.reshape(depth2, depth_shape)

    return image1,image2, annotation1,depth2,annotation2,depth1
