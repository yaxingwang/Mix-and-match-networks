"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import tensorflow as tf

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

def preprocess_annotation(annotation_batch,number_class):
	constant_255 = tf.constant(255,dtype = tf.int32)
	constant_class =tf.constant(number_class,dtype = tf.int32) 

	ones_tensors = tf.ones_like(annotation_batch,dtype = tf.int32) * (-1)
	mask_255 = tf.not_equal(annotation_batch, constant_255)
	annotation_batch = tf.where(mask_255,annotation_batch,ones_tensors)
	annotation_batch = tf.one_hot(annotation_batch,depth = constant_class)
        return annotation_batch
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def mask_object(annotation_batch,l0):
     

    constant_Bed = tf.constant(1,dtype = tf.int32)
    Bed_mask = tf.equal(annotation_batch,constant_Bed)
    Bed_mask = tf.to_float(Bed_mask)
    Bed_mask_num = tf.reduce_sum(Bed_mask)



    constant_books = tf.constant(2,dtype = tf.int32)
    books_mask = tf.equal(annotation_batch,constant_books)
    books_mask = tf.to_float(books_mask) 
    books_mask_num = tf.reduce_sum(books_mask)

    constant_Ceiling = tf.constant(3,dtype = tf.int32)
    Ceiling_mask = tf.equal(annotation_batch,constant_Ceiling)
    Ceiling_mask = tf.to_float(Ceiling_mask) 
    Ceiling_mask_num = tf.reduce_sum(Ceiling_mask)

    constant_chair = tf.constant(4,dtype = tf.int32)
    chair_mask = tf.equal(annotation_batch,constant_chair)
    chair_mask = tf.to_float(chair_mask) 
    chair_mask_num = tf.reduce_sum(chair_mask)

    constant_Furniture = tf.constant(6,dtype = tf.int32)
    Furniture_mask = tf.equal(annotation_batch,constant_Furniture)
    Furniture_mask = tf.to_float(Furniture_mask) 
    Furniture_mask_num = tf.reduce_sum(Furniture_mask)

    constant_Objects = tf.constant(7,dtype = tf.int32)
    Objects_mask = tf.equal(annotation_batch,constant_Objects)
    Objects_mask = tf.to_float(Objects_mask) 
    Objects_mask_num = tf.reduce_sum(Objects_mask)

    constant_Picture = tf.constant(8,dtype = tf.int32)
    Picture_mask = tf.equal(annotation_batch,constant_Picture)
    Picture_mask = tf.to_float(Picture_mask) 
    Picture_mask_num = tf.reduce_sum(Picture_mask)

    constant_sofa = tf.constant(9,dtype = tf.int32)
    Sofa_mask = tf.equal(annotation_batch,constant_sofa)
    Sofa_mask = tf.to_float(Sofa_mask) 
    Sofa_mask_num = tf.reduce_sum(Sofa_mask)

    constant_Table = tf.constant(10,dtype = tf.int32)
    Table_mask = tf.equal(annotation_batch,constant_Table)
    Table_mask = tf.to_float(Table_mask) 
    Table_mask_num = tf.reduce_sum(Table_mask)

    constant_TV = tf.constant(11,dtype = tf.int32)
    TV_mask = tf.equal(annotation_batch,constant_TV)
    TV_mask = tf.to_float(TV_mask) 
    TV_mask_num = tf.reduce_sum(TV_mask)

    constant_Window = tf.constant(13,dtype = tf.int32)
    Window_mask = tf.equal(annotation_batch,constant_Window)
    Window_mask = tf.to_float(Window_mask) 
    Window_mask_num = tf.reduce_sum(Window_mask)

    mask = Bed_mask +  books_mask + Ceiling_mask +   chair_mask  + Furniture_mask +  Objects_mask +  Picture_mask  + Sofa_mask + Table_mask+TV_mask+ Window_mask
    valid_value =Window_mask_num + TV_mask_num  + Table_mask_num  + Sofa_mask_num  + Picture_mask_num  + Objects_mask_num  + Furniture_mask_num  + chair_mask_num  + Ceiling_mask_num  + books_mask_num  + Bed_mask_num 
    l0 = tf.squeeze(l0)

    l = tf.multiply(mask,l0) 
    l = tf.expand_dims(l, axis = 3)
    return l,valid_value 

