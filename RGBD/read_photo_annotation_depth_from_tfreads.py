from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)
import tensorflow as tf

def read_photo_annotation_depth(batch_size = 1):
	def convert2float(image):
  	 	""" Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
 		"""
  		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	        return (image/127.5) - 1.0
	image_train_size = [256, 256]
	number_of_classes = 21

#	tfrecord_filename = '/home/yaxing/DATA/scene_net_augmented_train_200.tfrecords'
	tfrecord_filename = '/home/yaxing/data/scene5M/scene_net_augmented_train_16k.tfrecords'

	pascal_voc_lut ={0:'Unknown',1:'Bed',2:'Books',3:'Ceiling',4:'Chair',5:'Floor',6:'Furniture',7:'Objects',8:'Picture',9:'Sofa',10:'Table',11:'TV',12:'Wall',13:'Window' }
	class_labels = pascal_voc_lut.keys()

	filename_queue = tf.train.string_input_producer(
	    [tfrecord_filename], num_epochs=50000)

	image1, image2,annotation1,depth2,annotation2,depth1 = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

	# Various data augmentation stages
	image1, annotation1,depth2,image2,annotation2,depth1 = flip_randomly_left_right_image_with_annotation(image1, annotation1,depth2,image2,annotation2,depth1)

	# image = distort_randomly_image_color(image)

	resized_image1, resized_annotation1, resized_depth2,resized_image2,resized_annotation2, resized_depth1 = scale_randomly_image_with_annotation_with_fixed_size_output(image1, annotation1, depth2,image_train_size,img_tensor2=image2,annotation_tensor2 = annotation2,depth_tensor1= depth1)


	resized_annotation1 = tf.squeeze(resized_annotation1)
	resized_annotation2 = tf.squeeze(resized_annotation2)
	image_batch1, annotation_batch1,depth_batch2,image_batch2, annotation_batch2,depth_batch1 = tf.train.shuffle_batch( [resized_image1, resized_annotation1,resized_depth2,resized_image2, resized_annotation2, resized_depth1],
						     batch_size=batch_size,
						     capacity=3000,
						     num_threads=2,
						     min_after_dequeue=1000)
	return tf.to_float(image_batch1) / 127.5 - 1., annotation_batch1, (tf.exp(depth_batch1) - 1.)/ 1000., tf.to_float(image_batch2) / 127.5 - 1., annotation_batch2, (tf.exp(depth_batch2) - 1.)/ 1000.

