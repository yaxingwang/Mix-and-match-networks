
# coding: utf-8

# In[1]:

import os, sys
from PIL import Image
import pdb
sys.path.append("/home/yaxing/softes/tensorflow_FCN/tf-image-segmentation/")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

pascal_root = '/home/yaxing/softes/tensorflow_FCN/dataset/VOC2012'
pascal_berkeley_root = '/home/yaxing/softes/tensorflow_FCN/dataset/benchmark_RELEASE'

from tf_image_segmentation.utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from tf_image_segmentation.utils.pascal_voc import convert_pascal_berkeley_augmented_mat_annotations_to_png
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord
# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs =                 get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,                                                                                                                                                  
                pascal_berkeley_root=pascal_berkeley_root,
                mode=2)

overall_train_image_annotation_filename_pairs = []
for i in xrange(1000):
  for j in xrange(1,51):
    labels_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/semantic_class/semantic_class_%d_%d.png'%(i,j*25)
# You can create your own tfrecords file by providing
    images_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/val/0/%d/photo/%d.jpg'%(i,j*25)
    depth_path = '/home/yaxing/softes/pySceneNetRGBD-master/data/val/0/%d/depth/%d.png'%(i,j*25)
    image_label_pairs = (images_path,labels_path,depth_path)
    overall_train_image_annotation_filename_pairs.append(image_label_pairs)



write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='./dataset/scene_net_augmented_train.tfrecords')




