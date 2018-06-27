from __future__ import division
import matplotlib
matplotlib.use('Agg')

import os
import pdb
import time
import logging
import numpy as np
import tensorflow as tf


from ops import *
from utils import *
from glob import glob
from six.moves import xrange
from scipy.misc import imsave
from layers import unpool_with_argmax
from vgg16_first_5_layers.five_layers import vgg_enconder_only_one
from tf_image_segmentation.utils.training import get_valid_logits_and_labels
from read_photo_annotation_depth_from_tfreads import read_photo_annotation_depth
from tf_image_segmentation.utils.visualization import visualize_raw_image_synth_images 

vgg_enconder_only_one_init= vgg_enconder_only_one()
slim = tf.contrib.slim
logging.basicConfig(level = logging.INFO)

class mmnet(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L2_lambda = 10, anno_w = 100, dept_w = 10, auto_w = 10, imag_w = 10, feat_w = 10,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None,output_c_dim_annotation = 14,output_c_dim_depth = 1, REAL_LABEL = 1, number_class = 14 ):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.REAL_LABEL = REAL_LABEL
        self.number_class = number_class

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
	self.output_c_dim_depth = output_c_dim_depth 
	self.output_c_dim_annotation = output_c_dim_annotation 


        self.anno_w = anno_w
        self.dept_w = dept_w
        self.auto_w = auto_w 
        self.imag_w = imag_w 
        self.feat_w = feat_w 
        self.L2_lambda = L2_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.build_batchnorm()

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
	self.pascal_voc_lut ={0:'None',1:'Bed',2:'Books',3:'Ceiling',4:'Chair',5:'Floor',6:'Furniture',7:'Objects',8:'Picture',9:'Sofa',10:'Table',11:'TV',12:'Wall',13:'Window',255:'Unknown'}
	self.class_labels = self.pascal_voc_lut.keys()
        self.build_model_annotation()
        #self.build_model()

    def build_model_annotation(self):

        self.para_anno_dep = tf.placeholder(tf.float32, name='para_anno_dep')
        self.noise_features = tf.placeholder(tf.float32,shape = (self.batch_size, self.output_size/32,self.output_size/32,512), name='noise')

        # Reading data 
        # Image is [-1, 1], annotation is label(0 or 1), most of value of depth are less than 10 
	image1_batch, annotation1_batch, depth1_batch, image2_batch, annotation2_batch, depth2_batch = read_photo_annotation_depth(self.batch_size)

        # Photo 
        # For visualization during train
	self.image1_batch_tensor = image1_batch
	self.image2_batch_tensor = image2_batch

        # Annotation 
        # For visualization
	self.annotation1_batch_tensor = annotation1_batch 
	self.annotation2_batch_tensor = annotation2_batch 

        # Depth 
        # For visualization
	self.depth1_batch_tensor = depth1_batch 
	self.depth2_batch_tensor = depth2_batch 
	# Photo domain ---> annotation domain
	image1_pooling3_4_5, image1_indexs_max_pooling = self.image2annotation_encoder(image = image1_batch,name = 'G', input_name = 'photo') 
	logit = self.image2annotation_decoder(features = image1_pooling3_4_5[-1],indexs_max_pooling = image1_indexs_max_pooling, name = 'G', noise_features = self.noise_features) 
        # cross-entroy
        cross_entropy_from_image1, pred, probabilities =  self.annotation_loss(annotation = annotation1_batch, logit = logit)
        self.cross_entropy_from_image1 = cross_entropy_from_image1
        self.pred_train = pred
        self.probabilities_train = probabilities
        self.annotation_cross_from_image1_tv = tf.summary.scalar("cross_entropy_from_image1", cross_entropy_from_image1)



	# photo domain --- depth domain
	image2_pooling3_4_5, image2_indexs_max_pooling = self.image2annotation_encoder(image = image2_batch, name = 'G', input_name = 'photo', reuse = True) # Encoder of image is same
	fake_depth_from_image2 = self.image2depth_decoder(features = image2_pooling3_4_5[-1],indexs_max_pooling = image2_indexs_max_pooling, name = 'G', noise_features = self.noise_features) 


        depth_loss_from_image2 =  self.depth_loss(depth2_batch,fake_depth=fake_depth_from_image2)
	self.pred_depth_from_image2 = fake_depth_from_image2
        self.depth_loss_from_image2_tv = tf.summary.scalar("depth_loss_from_image2", depth_loss_from_image2)
 
	# annotation domain --- photo domain
	self.annotation1_batch_expand_channels = preprocess_annotation(annotation1_batch, self.number_class)
        annotaion1_pooling3_4_5, annotaion1_indexs_max_pooling = self.annotation2image_encoder(image = self.annotation1_batch_expand_channels, name = 'F', input_name = 'annotation')
        fake_photo_from_real_annotation1 = self.annotation2image_decoder(features = annotaion1_pooling3_4_5[-1], name = 'F')
        self.fake_photo_from_real_annotation1 = fake_photo_from_real_annotation1

        # annotation domain --- annotation domain
	logit_from_real_annotation1 = self.image2annotation_decoder(features = annotaion1_pooling3_4_5[-1], indexs_max_pooling = annotaion1_indexs_max_pooling, name = 'G', reuse = True, using_noise = False) 
        cross_entropy_from_real_annotation1, _, _ =  self.annotation_loss(annotation = annotation1_batch, logit = logit_from_real_annotation1)
        self.annotation_cross_from_annotation1_tv  = tf.summary.scalar("cross_entropy_from_annotation1", cross_entropy_from_real_annotation1)
 
	# depth domain --- photo domain
        depth2_pooling3_4_5, depth2_indexs_max_pooling = self.depth2image_encoder(image = depth2_batch, name = 'F', input_name = 'depth')
        fake_photo_from_real_depth2 = self.annotation2image_decoder(features = depth2_pooling3_4_5[-1], name = 'F', reuse = True)# decoder is same 
 
        self.fake_photo_from_real_depth2 = fake_photo_from_real_depth2
	# depth domain --- depth domain
	fake_depth_from_real_depth2 = self.image2depth_decoder(features = depth2_pooling3_4_5[-1], indexs_max_pooling = depth2_indexs_max_pooling, name = 'G', reuse = True, using_noise = False) 
        depth_loss_from_real_depth2 =  self.depth_loss(depth2_batch, fake_depth= fake_depth_from_real_depth2)
        self.depth_loss_from_depth2_tv = tf.summary.scalar("depth_loss_from_depth2", depth_loss_from_real_depth2)

  
	# photo domain --- photo domain
        # image1
        fake_photo_from_image1 = self.annotation2image_decoder(features = image1_pooling3_4_5[-1], name = 'F', reuse = True)# decoder is same 
        # image2
        fake_photo_from_image2 = self.annotation2image_decoder(features = image2_pooling3_4_5[-1], name = 'F', reuse = True)# decoder is same 
        # adversarial losses for image
        # annoation
        L1_real_annotation1, G_real_loss_real_annotation1, D_loss_real_annotation1 =self.G_D_loss(real = image1_batch,fake = fake_photo_from_real_annotation1, reuse = False) 
        # depth 
        L1_real_depth2, G_loss_real_depth2, D_loss_real_depth2 =self.G_D_loss(real = image2_batch,fake = fake_photo_from_real_depth2) 
        # image1
        L1_real_image1, G_loss_real_image1, D_loss_image1 =self.G_D_loss(real = image1_batch,fake = fake_photo_from_image1) 
        L1_real_image2, G_loss_real_image2, D_loss_image2 =self.G_D_loss(real = image2_batch,fake = fake_photo_from_image2) 

        # distance for images
        L1 = tf.reduce_mean(L1_real_annotation1 + L1_real_depth2 + L1_real_image1 + L1_real_image2)
        # G loss 
        G_loss = G_real_loss_real_annotation1 + G_loss_real_depth2 + G_loss_real_image1 + G_loss_real_image2  
        D_loss = D_loss_real_annotation1 + D_loss_real_depth2 + D_loss_image1 + D_loss_image2 

	self.L1_image1_image2_tv = tf.summary.scalar("L1_image1_image2", L1_real_image1 + L1_real_image2)
	self.G_L1_loss_tv = tf.summary.scalar("L1_loss", L1)
	self.G_loss_tv = tf.summary.scalar("G_loss", G_loss)
	self.D_loss_tv = tf.summary.scalar("D_loss", D_loss)
        # Latest space alignment


	feature_loss_image1 = self.feature_loss(image1_pooling3_4_5[0], image1_pooling3_4_5[1], image1_pooling3_4_5[2], annotaion1_pooling3_4_5[0], annotaion1_pooling3_4_5[1], annotaion1_pooling3_4_5[2])
	feature_loss_image2 = self.feature_loss(image2_pooling3_4_5[0], image2_pooling3_4_5[1], image2_pooling3_4_5[2], depth2_pooling3_4_5[0], depth2_pooling3_4_5[1], depth2_pooling3_4_5[2])

	self.image1_annotation1_feature_loss_tv = tf.summary.scalar("image1_annotation1_feature_loss_tv", feature_loss_image1)
	self.image2_depth2_feature_loss_tv = tf.summary.scalar("image2_depth2_feature_loss_tv", feature_loss_image2)

	self.pooling5_image2_tv = tf.summary.histogram('pooling5_image2', image2_pooling3_4_5[2])
	self.pooling5_depth2_tv = tf.summary.histogram('pooling5_depth2', depth2_pooling3_4_5[2])

	self.pooling4_image2_tv = tf.summary.histogram('pooling4_image2', image2_pooling3_4_5[1])
	self.pooling4_depth2_tv = tf.summary.histogram('pooling4_depth2', depth2_pooling3_4_5[1])

	self.pooling3_image2_tv = tf.summary.histogram('pooling3_image2', image2_pooling3_4_5[0])
	self.pooling3_depth2_tv = tf.summary.histogram('pooling3_depth2', depth2_pooling3_4_5[0])



	self.pooling5_image1_tv = tf.summary.histogram('pooling5_image1', image1_pooling3_4_5[2])
	self.pooling5_annotation1_tv = tf.summary.histogram('pooling5_annotation1', annotaion1_pooling3_4_5[2])

	self.pooling4_image1_tv = tf.summary.histogram('pooling4_image1', image1_pooling3_4_5[1])
	self.pooling4_annotation1_tv = tf.summary.histogram('pooling4_annotation1', annotaion1_pooling3_4_5[1])

	self.pooling3_image1_tv = tf.summary.histogram('pooling3_image1', image1_pooling3_4_5[0])
	self.pooling3_annotation1_tv = tf.summary.histogram('pooling3_annotation1', annotaion1_pooling3_4_5[0])

        # Max values of feature
	max_feature_loss_image1 = self.max_feature_map_loss(image1_pooling3_4_5[2], annotaion1_pooling3_4_5[2])
	max_feature_loss_image2 = self.max_feature_map_loss(image2_pooling3_4_5[2], depth2_pooling3_4_5[2])

	self.pooling5_image1_max_tv = tf.summary.scalar("pooling5_image1_annotation1_max", max_feature_loss_image1)
	self.pooling5_image2_max_tv = tf.summary.scalar("pooling5_image2_depth2_max", max_feature_loss_image2)
#####################################################################################################################################################################################################
		 	  	  
        
        self.g =  self.anno_w * cross_entropy_from_image1\
                + self.dept_w * depth_loss_from_image2\
                + self.auto_w * (cross_entropy_from_real_annotation1  + depth_loss_from_real_depth2)\
                + self.feat_w * (L1   + G_loss) \
                + self.feat_w * (feature_loss_image1 +  feature_loss_image2  + max_feature_loss_image1 + max_feature_loss_image2)
        self.d = self.imag_w * D_loss

        # visualization dur train.Note we use pooling indice from image1. 
	logits_from_depth2 = self.image2annotation_decoder(features = depth2_pooling3_4_5[2], indexs_max_pooling = image1_indexs_max_pooling, name = 'G',reuse = True, using_noise = False) 

	self.pred_test = tf.argmax(logits_from_depth2, dimension=3)
	self.probabilities_test = tf.nn.softmax(logits_from_depth2)
	
        t_vars = tf.trainable_variables()

        G_vars = [var for var in t_vars if 'G/' in var.name]
        F_vars = [var for var in t_vars if 'F/' in var.name]
        self.g_vars = G_vars + F_vars
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]


    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g, var_list=self.g_vars)

        init_op =tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op) 
        g_sum = self.summary_g()
        d_sum = tf.summary.merge([self.D_loss_tv])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        pdb.set_trace()
        self.saver = tf.train.Saver(self.g_vars)
	self.saver.restore(self.sess, "./pretrain/pix2pix.model-215001")
        self.saver = tf.train.Saver(max_to_keep=100)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = self.sess,coord = coord)

        counter = 1
	Inter_D = 1
	try:
	   while not coord.should_stop():
                # Update D network
                if np.mod(Inter_D, 2) == 1:
                	_, summary_str = self.sess.run([d_optim, d_sum],
                				       feed_dict={self.noise_features:np.abs(np.random.normal(0, 0.5, (self.batch_size, 8, 8, 512)))})
                	self.writer.add_summary(summary_str, counter)
                
                # Update G network
                _, summary_str, cross_entropy_from_image1 = self.sess.run([g_optim, g_sum, self.cross_entropy_from_image1], feed_dict={ self.noise_features:np.abs(np.random.normal(0, 0.5, (self.batch_size, 8, 8, 512)))})
                self.writer.add_summary(summary_str, counter)
                self.writer.flush()
                print ('Current annotation loss: ' + str(cross_entropy_from_image1))
                
                Inter_D +=1
                counter += 1
                if np.mod(counter, 100) == 2:
                    self.visua(counter)
                # train loss for annotation
                if np.mod(counter, 10000) == 1:
                	self.save(args.checkpoint_dir, counter)

	except KeyboardInterrupt:
		logging.info('Interrupted')
		coord.request_stop()
	except Exception as e:
		coord.request_stop(e)

	finally:
                #self.save(args.checkpoint_dir, counter)
		logging.info('Model saved in file')
		coord.request_stop()
		coord.join(threads)
    
    def discriminator(self, image, y=None, reuse=False,use_lsgan = False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)

            #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return (h3, h3) if use_lsgan else (tf.nn.sigmoid(h3), h3)


    def annotation2image_encoder(self, image, name = 'F', input_name = 'annotation', encoder_name ='twin_encoder_1', reuse = False):

        with tf.variable_scope(name + '/en/' + input_name, reuse = reuse) as scope:
		net = tf.nn.relu(self.g_bn_e1_0(conv2d(image, self.gf_dim,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name=input_name + '/'+ 'g_e1_conv_0')))
		net = tf.nn.relu(self.g_bn_e1_1(conv2d(net, self.gf_dim,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name=input_name + '/'+ 'g_e1_conv_1')))
                net, arg1 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
	# e1 is (128 x 128 x self.gf_dim)
	with tf.variable_scope(name + '/' + encoder_name, reuse = reuse) as scope:
		net = tf.nn.relu(self.g_bn_e2_0(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e2_conv_0')))
		net = tf.nn.relu(self.g_bn_e2_1(conv2d(net, self.gf_dim*2,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e2_conv_1')))
		net, arg2 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
		# e2 is (64 x 64 x self.gf_dim*2)
		net =tf.nn.relu(self.g_bn_e3_0(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_0')))
		net =tf.nn.relu(self.g_bn_e3_1(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_1')))
		net =tf.nn.relu(self.g_bn_e3_2(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_2')))
		net, arg3 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool3')
		pooling3 = net
		# e3 is (32 x 32 x self.gf_dim*4)
		net = tf.nn.relu(self.g_bn_e4_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_0')))
		net = tf.nn.relu(self.g_bn_e4_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_1')))
		net = tf.nn.relu(self.g_bn_e4_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_2')))
		net, arg4 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
		pooling4 = net
		# e4 is (16 x 16 x self.gf_dim*8)
		net = tf.nn.relu(self.g_bn_e5_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_0')))
		net = tf.nn.relu(self.g_bn_e5_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_1')))
		net = tf.nn.relu(self.g_bn_e5_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_2')))
		net, arg5 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool5')
		pooling5 = net
		# e5 is (8 x 8 x self.gf_dim*16)
		pooling3_4_5 = (pooling3, pooling4, pooling5) 
		indexs_max_pooling = (arg1, arg2, arg3, arg4, arg5)
        return pooling3_4_5, indexs_max_pooling
    def depth2image_encoder(self, image, name = 'F', input_name = 'depth', encoder_name ='twin_encoder_2', reuse = False):

        with tf.variable_scope(name + '/en/' + input_name, reuse = reuse) as scope:
		net = tf.nn.relu(self.g_bn_e1_0(conv2d(image, self.gf_dim,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name=input_name + '/'+ 'g_e1_conv_0')))
		net = tf.nn.relu(self.g_bn_e1_1(conv2d(net, self.gf_dim,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name=input_name + '/'+ 'g_e1_conv_1')))
                net, arg1 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
	# e1 is (128 x 128 x self.gf_dim)
	with tf.variable_scope(name + '/' + encoder_name, reuse = reuse) as scope:
		net = tf.nn.relu(self.g_bn_e2_0(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e2_conv_0')))
		net = tf.nn.relu(self.g_bn_e2_1(conv2d(net, self.gf_dim*2,k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e2_conv_1')))
		net, arg2 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
		# e2 is (64 x 64 x self.gf_dim*2)
		net =tf.nn.relu(self.g_bn_e3_0(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_0')))
		net =tf.nn.relu(self.g_bn_e3_1(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_1')))
		net =tf.nn.relu(self.g_bn_e3_2(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e3_conv_2')))
		net, arg3 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool3')
		pooling3 = net
		# e3 is (32 x 32 x self.gf_dim*4)
		net = tf.nn.relu(self.g_bn_e4_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_0')))
		net = tf.nn.relu(self.g_bn_e4_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_1')))
		net = tf.nn.relu(self.g_bn_e4_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e4_conv_2')))
		net, arg4 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
		pooling4 = net
		# e4 is (16 x 16 x self.gf_dim*8)
		net = tf.nn.relu(self.g_bn_e5_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_0')))
		net = tf.nn.relu(self.g_bn_e5_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_1')))
		net = tf.nn.relu(self.g_bn_e5_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_e5_conv_2')))
		net, arg5 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool5')
		pooling5 = net
		# e5 is (8 x 8 x self.gf_dim*16)
		pooling3_4_5 = (pooling3, pooling4, pooling5) 
		indexs_max_pooling = (arg1, arg2, arg3, arg4, arg5)
        return pooling3_4_5, indexs_max_pooling

    def annotation2image_decoder(self, features, name = 'F', output_domain = 'photo', decoder_name='twin_decoder_0', noise_features = None, reuse = False):

        net = features  
	with tf.variable_scope(name + '/' + decoder_name,reuse = reuse) as scope:
            net, _, _ = deconv2d(tf.nn.relu(net),
                [self.batch_size, 16, 16, self.gf_dim*8], name='g_d4', with_w=True)
            net = self.g_bn_d4_0(net)

            net, _, _ = deconv2d(tf.nn.relu(net),
                [self.batch_size, 32, 32, self.gf_dim*4], name='g_d5', with_w=True)
            net = self.g_bn_d5_0(net)

	    net, _, _ = deconv2d(tf.nn.relu(net),
		[self.batch_size, 64, 64, self.gf_dim*2], name='g_d6', with_w=True)
	    net = self.g_bn_d6_0(net)

	    net, _, _ = deconv2d(tf.nn.relu(net),
		[self.batch_size, 128, 128, self.gf_dim], name='g_d7', with_w=True)
	    net = self.g_bn_d7_0(net)

        with tf.variable_scope(name + '/de/' + output_domain,reuse = reuse) as scope:
            net, _, _ = deconv2d(tf.nn.relu(net),
                [self.batch_size, 256, 256, self.output_c_dim], name=output_domain + '/'+'g_d8', with_w=True)
            output = tf.nn.tanh(net)

        return output 



    def image2annotation_encoder(self, image, name = 'G',input_name = 'photo', reuse = False):
        # Using vgg pretrained network 
        with tf.variable_scope(name + '/en/' + input_name,reuse = reuse) as scope:
		pooling3_4_5, indexs_max_pooling = vgg_enconder_only_one_init.build(image,reuse = reuse)
        return pooling3_4_5, indexs_max_pooling  


    def image2annotation_decoder(self, features, indexs_max_pooling, name = 'G', output_domain = 'annotation',  decoder_name='twin_decoder_1', noise_features = None, reuse = False, using_noise = True):

        net = features
	arg1,arg2,arg3,arg4,arg5 = indexs_max_pooling   

	with tf.variable_scope(name + '/' + decoder_name,reuse = reuse) as scope:
            if using_noise:
	        net = net + noise_features
            net = unpool_with_argmax(net,arg5,name = 'maxunpool5')
            net = tf.nn.relu(self.g_bn_d5_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_0')))
            net = tf.nn.relu(self.g_bn_d5_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_1')))
            net = tf.nn.relu(self.g_bn_d5_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_2')))
	    # now net  is (16 x 16 x self.gf_dim*16)

            net = unpool_with_argmax(net,arg4,name = 'maxunpool4')
            net = tf.nn.relu(self.g_bn_d4_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d4_conv_0')))
            net = tf.nn.relu(self.g_bn_d4_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d4_conv_1')))
            net = tf.nn.relu(self.g_bn_d4_2(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1,  name='g_d4_conv_2')))
	    # now net  is (32 x 32 x self.gf_dim*8)

            net = unpool_with_argmax(net,arg3,name = 'maxunpool3')
            net = tf.nn.relu(self.g_bn_d3_0(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_0')))
            net = tf.nn.relu(self.g_bn_d3_1(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_1')))
            net = tf.nn.relu(self.g_bn_d3_2(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_2')))
	    # now net  is (64 x 64 x self.gf_dim*4)

            net = unpool_with_argmax(net,arg2,name = 'maxunpool2')
            net = tf.nn.relu(self.g_bn_d2_0(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d2_conv_0')))
            net = tf.nn.relu(self.g_bn_d2_1(conv2d(net, self.gf_dim, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d2_conv_1')))
	    # now net  is (128 x 128 x self.gf_dim*2)

        with tf.variable_scope(name + '/de/' + output_domain,reuse = reuse) as scope:
            net = unpool_with_argmax(net,arg1,name = 'maxunpool1')
            net = tf.nn.relu(self.g_bn_d1_0(conv2d(net, self.gf_dim, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d1_conv_0')))
            output  = tf.nn.relu(self.g_bn_d1_1(conv2d(net, self.output_c_dim_annotation, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d1_conv_1')))
	    # now net  is (256 x 256 x self.gf_dim*2)
		    
        return  output 

    def image2depth_decoder(self, features, indexs_max_pooling, name = 'G', output_domain = 'depth', decoder_name='twin_decoder_2', noise_features = None, reuse = False, using_noise = True):

        net = features
	arg1,arg2,arg3,arg4,arg5 = indexs_max_pooling   

	with tf.variable_scope(name + '/' + decoder_name,reuse = reuse) as scope:
            if using_noise:
	        net = net + noise_features
            net = unpool_with_argmax(net,arg5,name = 'maxunpool5')
            net = tf.nn.relu(self.g_bn_d5_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_0')))
            net = tf.nn.relu(self.g_bn_d5_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_1')))
            net = tf.nn.relu(self.g_bn_d5_2(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d5_conv_2')))
	    # now net  is (16 x 16 x self.gf_dim*16)

            net = unpool_with_argmax(net,arg4,name = 'maxunpool4')
            net = tf.nn.relu(self.g_bn_d4_0(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d4_conv_0')))
            net = tf.nn.relu(self.g_bn_d4_1(conv2d(net, self.gf_dim*8, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d4_conv_1')))
            net = tf.nn.relu(self.g_bn_d4_2(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1,  name='g_d4_conv_2')))
	    # now net  is (32 x 32 x self.gf_dim*8)

            net = unpool_with_argmax(net,arg3,name = 'maxunpool3')
            net = tf.nn.relu(self.g_bn_d3_0(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_0')))
            net = tf.nn.relu(self.g_bn_d3_1(conv2d(net, self.gf_dim*4, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_1')))
            net = tf.nn.relu(self.g_bn_d3_2(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d3_conv_2')))
	    # now net  is (64 x 64 x self.gf_dim*4)

            net = unpool_with_argmax(net,arg2,name = 'maxunpool2')
            net = tf.nn.relu(self.g_bn_d2_0(conv2d(net, self.gf_dim*2, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d2_conv_0')))
            net = tf.nn.relu(self.g_bn_d2_1(conv2d(net, self.gf_dim, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d2_conv_1')))
	    # now net  is (128 x 128 x self.gf_dim*2)

        with tf.variable_scope(name + '/de/' + output_domain,reuse = reuse) as scope:
            net = unpool_with_argmax(net,arg1,name = 'maxunpool1')
            net = tf.nn.relu(self.g_bn_d1_0(conv2d(net, self.gf_dim, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d1_conv_0')))
            output  = conv2d(net, self.output_c_dim_depth, k_h = 3,k_w = 3,d_h = 1, d_w = 1, name='g_d1_conv_1')
	    # now net  is (256 x 256 x self.gf_dim*2)
        return  output 
    def build_batchnorm(self):
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e1_0 = batch_norm(name='g_bn_e1_0')
        self.g_bn_e1_1 = batch_norm(name='g_bn_e1_1')
        self.g_bn_e2_0 = batch_norm(name='g_bn_e2_0')
        self.g_bn_e2_1 = batch_norm(name='g_bn_e2_1')
        self.g_bn_e3_0 = batch_norm(name='g_bn_e3_0')
        self.g_bn_e3_1 = batch_norm(name='g_bn_e3_1')
        self.g_bn_e3_2 = batch_norm(name='g_bn_e3_2')
        self.g_bn_e4_0 = batch_norm(name='g_bn_e4_0')
        self.g_bn_e4_1 = batch_norm(name='g_bn_e4_1')
        self.g_bn_e4_2 = batch_norm(name='g_bn_e4_2')
        self.g_bn_e5_0 = batch_norm(name='g_bn_e5_0')
        self.g_bn_e5_1 = batch_norm(name='g_bn_e5_1')
        self.g_bn_e5_2 = batch_norm(name='g_bn_e5_2')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1_0 = batch_norm(name='g_bn_d1_0')
        self.g_bn_d1_1 = batch_norm(name='g_bn_d1_1')
        self.g_bn_d2_0 = batch_norm(name='g_bn_d2_0')
        self.g_bn_d2_1 = batch_norm(name='g_bn_d2_1')
        self.g_bn_d3_0 = batch_norm(name='g_bn_d3_0')
        self.g_bn_d3_1 = batch_norm(name='g_bn_d3_1')
        self.g_bn_d3_2 = batch_norm(name='g_bn_d3_2')
        self.g_bn_d4_0 = batch_norm(name='g_bn_d4_0')
        self.g_bn_d4_1 = batch_norm(name='g_bn_d4_1')
        self.g_bn_d4_2 = batch_norm(name='g_bn_d4_2')
        self.g_bn_d5_0 = batch_norm(name='g_bn_d5_0')
        self.g_bn_d5_1 = batch_norm(name='g_bn_d5_1')
        self.g_bn_d5_2 = batch_norm(name='g_bn_d5_2')
        self.g_bn_d6_0 = batch_norm(name='g_bn_d6')
        self.g_bn_d7_0 = batch_norm(name='g_bn_d7')

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    def G_D_loss(self,real,fake,reuse = True):
            D, D_logits = self.discriminator(real, reuse=reuse,use_lsgan=True)
            D_, D_logits_ = self.discriminator(fake, reuse=True,use_lsgan=True)

            L1 = self.L2_lambda * tf.reduce_mean(tf.square(real - fake))
            G_loss = self.generator_loss(D_, use_lsgan=True)  
            D_loss = self.discriminator_loss(D, D_, use_lsgan=True)
            return L1,G_loss,D_loss
    def discriminator_loss(self,D_true, D_fake, use_lsgan=True):
	    """ Note: default: D(y).shape == (batch_size,5,5,1),
			       fake_buffer_size=50, batch_size=1
	    Args:
	      G: generator object
	      D: discriminator object
	      y: 4D tensor (batch_size, image_size, image_size, 3)
	    Returns:
	      loss: scalar
	    """
	    if use_lsgan:
	      # use mean squared error
	      error_real = tf.reduce_mean(tf.squared_difference(D_true, self.REAL_LABEL))
	      error_fake = tf.reduce_mean(tf.square(D_fake))
	    else:
	      # use cross entropy
	      error_real = -tf.reduce_mean(safe_log(D_true))
	      error_fake = -tf.reduce_mean(safe_log(1-D_fake))
	    loss = (error_real + error_fake) / 2
	    return loss
    def generator_loss(self,D_fake, use_lsgan=True):
	    """  fool discriminator into believing that G(x) is real
	    """
	    if use_lsgan:
	      # use mean squared error
	      loss = tf.reduce_mean(tf.squared_difference(D_fake, self.REAL_LABEL))
	    else:
	      # heuristic, non-saturating loss
	      loss = -tf.reduce_mean(safe_log(D_fake)) / 2
	    return loss

    def annotation_loss(self, annotation, logit):

        valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=annotation,logits_batch_tensor=logit, class_labels=self.class_labels) 
        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,\
                        labels=valid_labels_batch_tensor)
        # Normalize the cross entropy -- the number of elements
        # is different during each step due to mask out regions
        cross_entropy_sum = tf.reduce_mean(cross_entropies)
        pred = tf.argmax(logit, dimension=3)
        #self.pred = tf.expand_dims(pred, 3)
        probabilities = tf.nn.softmax(logit)

        return cross_entropy_sum, pred, probabilities 
    def depth_loss(self, depth_batch, fake_depth, Huber_loss = True):
 	    l0 = fake_depth  - depth_batch
            #l_mask,valid_num = mask_object(self.annotation_batch_tensor,l0)
            if Huber_loss:
		    l0 = tf.abs(l0)
		    c_max =tf.reduce_max(l0) 
		    l1 = tf.less_equal(l0, c_max)
		    l2 =  (tf.square(l0) + tf.square(c_max)) / (2 * c_max)
                    l = tf.reduce_mean(tf.where(l1, l0, l2)) 

	    else:

		    l1 = tf.reduce_mean(tf.square(l0)) 
		    l2 = self.depth_inside_lambda *tf.square(tf.reduce_mean(l0)) +self.depth_lambda * tf.reduce_mean(tf.abs(l0)) 
		    l = l1 + l2
	    return l 


    def max_feature_map_loss(self, feature_image_conv3,feature_modmain_conv3):

		image_feature_max = tf.reduce_sum(feature_image_conv3,axis = [1,2],keep_dims = True)
		feature_max = tf.reduce_max(image_feature_max)
		image_feature_max = tf.greater_equal(image_feature_max,image_feature_max/10.)
		
		image_feature_max = tf.to_float(image_feature_max)
		number_max = tf.reduce_sum(image_feature_max)
		max_value_image = tf.multiply(feature_image_conv3,image_feature_max)
		max_value_depth = tf.multiply(feature_modmain_conv3,image_feature_max)
		max_feature_loss = tf.reduce_sum(tf.square(max_value_image - max_value_depth))/ (self.batch_size*8*8*number_max)




		return max_feature_loss 
    def feature_loss(self,feature_image_conv3, feature_image_conv4, feature_image_conv5, feature_modmain_conv3, feature_modmain_conv4, feature_modmain_conv5):
		Feature_loss = tf.reduce_mean(tf.square(feature_image_conv3 - feature_modmain_conv3))\
			     + tf.reduce_mean(tf.square(feature_image_conv4 - feature_modmain_conv4))\
		             + tf.reduce_mean(tf.square(feature_image_conv5 - feature_modmain_conv5))
		return   Feature_loss


    def summary_g(self):
        return tf.summary.merge([
                        self.annotation_cross_from_image1_tv,
                        self.depth_loss_from_image2_tv,
                        self.annotation_cross_from_annotation1_tv, 
                        self.depth_loss_from_depth2_tv,
                    	self.L1_image1_image2_tv,
                    	self.G_L1_loss_tv, 
                    	self.G_loss_tv, 
                                    
                    	self.image1_annotation1_feature_loss_tv, 
                    	self.image2_depth2_feature_loss_tv, 
                    
                    	self.pooling5_image2_tv, 
                    	self.pooling5_depth2_tv, 
                    
                    	self.pooling4_image2_tv,
                    	self.pooling4_depth2_tv, 
                    
                    	self.pooling3_image2_tv, 
                    	self.pooling3_depth2_tv, 
                    
                    
                    	self.pooling5_image1_tv, 
                    	self.pooling5_annotation1_tv, 
                    
                    	self.pooling4_image1_tv,
                    	self.pooling4_annotation1_tv, 
                    
                    	self.pooling3_image1_tv, 
                    	self.pooling3_annotation1_tv, 

                    	self.pooling5_image1_max_tv, 
                    	self.pooling5_image2_max_tv
            ])


    def visua(self, counter):
            image1_batch, image2_batch, anno1_batch, anno2_batch, depth1_batch, depth2_batch, pred_anno_from_detph2, pred_anno_from_image1, pred_depth_from_image2, pred_image1_from_anno1, pred_image2_from_depth2\
                            = self.sess.run([self.image1_batch_tensor, self.image2_batch_tensor, \
                                             self.annotation1_batch_tensor, self.annotation2_batch_tensor,\
                                             self.depth1_batch_tensor, self.depth2_batch_tensor,\
                                             self.pred_test, self.pred_train,\
                                             self.pred_depth_from_image2,\
                                             self.fake_photo_from_real_annotation1, self.fake_photo_from_real_depth2],\
                                             feed_dict={ self.noise_features:np.abs(np.random.normal(0, 0.5,(self.batch_size, 8,8,512)))})
    
                    
    	    raw_image1, raw_image2 =np.squeeze((image1_batch[0] + 1.)*127.5), np.squeeze((image2_batch[0] + 1.)*127.5)
            raw_annotation1, raw_annotation2, pred_anno_from_test, pred_anno_from_train = np.squeeze(anno1_batch[0]), np.squeeze(anno2_batch[0]), np.squeeze(pred_anno_from_detph2[0]),np.squeeze(pred_anno_from_image1[0])
            raw_depth2, pred_depth = np.squeeze(depth2_batch[0]),np.squeeze(pred_depth_from_image2[0])
            pred_image_from_anno1, pred_image_from_depth2  = np.squeeze((pred_image1_from_anno1[0] + 1.)*127.5), np.squeeze((pred_image2_from_depth2[0] + 1.)*127.5)
    
            imsave('./visualization/pre_image.png', pred_image_from_anno1)
            imsave('./visualization/pre_image2.png', pred_image_from_depth2)
            imsave('./visualization/raw_image.png', raw_image1)
            imsave('./visualization/raw_image2.png', raw_image2)
            visualize_raw_image_synth_images(raw_image1, raw_depth2, pred_depth, pred_image_from_anno1, raw_image2, pred_image_from_depth2, (raw_annotation1, pred_anno_from_test, pred_anno_from_train, raw_annotation2), self.pascal_voc_lut, counter)
