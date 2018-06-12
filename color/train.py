import __future__

import os
import pdb
import logging
import numpy as np
import tensorflow as tf

from reader import Reader
from model import MmNet
from utils import ImagePool
from datetime import datetime
from utils import visualization
from data_labels import labels


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 128')
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
tf.flags.DEFINE_string('train_file', './dataset/train/domain_',
                       'tfrecords file for training, default: ./dataset/domain_11.tfrecords')
tf.flags.DEFINE_string('load_model','20171109-1200',
                        'folder of saved model that you wish to continue training (e.g. 20170725-1207), default: None')

def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass
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
            train_file = FLAGS.train_file)
      #G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x,fake_z,fake_x_from_z = cycle_gan.model()
        G_loss, D_Y_loss, F_loss, D_X_loss = MmNet.model()
        optimizers = MmNet.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)
        
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
            print('pretrained model is Done')
        else:
            sess.run(tf.global_variables_initializer())
            step = 0
        acount = 0
        domain_idx = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            fake_pool = [ImagePool(FLAGS.pool_size) for i in xrange(FLAGS.number_domain)]
            
            while not coord.should_stop():

                # Fake images which are corresponding to 11 domains
                # 'domain_idx' to select which domain except for anchor to synthesize anchor doamin image
                #Using loop to make 'domain_idx' increment and reset
                fake_ =[MmNet.loss[domain_idx][-1]] +  [MmNet.loss[i][-2] for i in xrange(FLAGS.number_domain-1)]
                # Output special fake images from current execution
                fake_gene= sess.run(fake_)
                # Building input dictionary for optimal function 
                feed_dict={MmNet.fake_set[i]: fake_pool[i].query(fake_gene[i]) for i in xrange(FLAGS.number_domain)}
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
              	sess.run([optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op], feed_dict))
                train_writer.add_summary(summary, step)
                train_writer.flush()
                domain_idx +=1
                if domain_idx ==FLAGS.number_domain-1:
                	domain_idx = 0
                if step % 100 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  G_loss   : {}'.format(G_loss_val))
                    logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info('  F_loss   : {}'.format(F_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                if step % 10000 == 0 and step > 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                step += 1
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
tf.app.run()
