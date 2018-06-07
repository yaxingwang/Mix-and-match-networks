import __future__

import os
import pdb
import logging
import numpy as np
import tensorflow as tf

from reader import Reader
from model import CycleGAN
from utils import ImagePool
from datetime import datetime
from utils import visualization
from lu_data_labels import labels


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 256')
tf.flags.DEFINE_integer('number_domain', 11, 'number of doamin, default: 11')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('lambda1', 10.0,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_float('lambda2', 10.0,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('test_file', './dataset/domain_',
                       'tfrecords file for training, default: ./dataset/domain_11.tfrecords')
tf.flags.DEFINE_string('saved_model','checkpoints/20171109-1200/model.ckpt-7703',
                        'folder of saved model that you want to restore), default: checkpoints/20171109-1200/model.ckpt-7703')

def test():
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda1,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf,
            number_domain=FLAGS.number_domain,
            train_file = FLAGS.test_file)
        G_loss, D_Y_loss, F_loss, D_X_loss = cycle_gan.model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(cycle_gan.F_train_var + cycle_gan.G_train_var)
        saver.restore(sess, FLAGS.saved_model)
        acount = 1
        domain_idx = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            fake_pool = [ImagePool(FLAGS.pool_size) for i in xrange(FLAGS.number_domain)]
            
            while not coord.should_stop():
                # get previously generated images
                #Probility = np.random.randint(2,size = 1)
                fake_ =[cycle_gan.loss[domain_idx][-1]] +  [cycle_gan.loss[i][-2] for i in xrange(FLAGS.number_domain-1)]
                fake_gene= sess.run(fake_)
                #fake_y_val, fake_x_val,fake_z_val,fake_x_from_z_val = sess.run([fake_y, fake_x, fake_z, fake_x_from_z])
                feed_dict={cycle_gan.fake_set[i]: fake_pool[i].query(fake_gene[i]) for i in xrange(FLAGS.number_domain)}
                raw_image_generated_images =sess.run(cycle_gan.raw_image_generated_images,feed_dict)
                pdb.set_trace()
                visualization(raw_image_generated_images,acount,labels)
                acount +=1
                domain_idx += 1
                print acount 
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
          #save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          #logging.info("Model saved in file: %s" % save_path)
          # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    test()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
tf.app.run()
