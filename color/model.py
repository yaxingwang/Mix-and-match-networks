import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
import pdb

from lu_data_labels import labels
REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self,
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10.0,
               lambda2=10.0,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=64,
	       number_domain = 4,
	       train_file = 'domain_'
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    self.use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.train_file =train_file 
    self.number_domain = number_domain

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    # Define two Generators which have different names
    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)

    # Define number_domain discriminators
    self.D_set = [Discriminator('D_%d'%i, self.is_training, norm=norm, use_sigmoid=self.use_sigmoid) for i in xrange(self.number_domain)]
    # Define number_domain fake samples  
    self.fake_set = [tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3]) for i in xrange(self.number_domain)]
  def model(self):

    # Import the compressed train datas
    self.reader_set =[Reader(self.train_file, name='%s/train_%d'%(labels[i],i),
	    image_size=self.image_size, batch_size=self.batch_size).feed() for i in xrange(self.number_domain)] 

    # Computing cycle losses for number_domain, which totally have number_domain - 1 items 
    self.cycle_loss_set = [self.cycle_consistency_loss(special_domain = i+1) for i in xrange(self.number_domain -1)]

    # Computing adversarial losses for number_domain  
    self.loss = [self.gan_cycle_loss(special_domain = i+1) for i in xrange(self.number_domain -1)]
    # Note that 'x' points to the anchor, 'y'
    self.G_y_set =sum([self.loss[i][0] for i in xrange(self.number_domain -1)])
    self.D_y_set =sum([self.loss[i][1] for i in xrange(self.number_domain -1)])
    self.F_x_set =sum([self.loss[i][2] for i in xrange(self.number_domain -1)])
    self.D_x_set =sum([self.loss[i][3] for i in xrange(self.number_domain -1)])#This is big question 
   # cycle_loss_twin = self.cycle_consistency_loss(self.G, self.F, x, z,enconder_name ='twin_enconder_1',deconder_name='twin_deconder_1')






 #   ################################################################################################################
    tf.summary.image('%s/generated_%s'%(labels[0],labels[1]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_0')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[2]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_1')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[3]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_2')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[4]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_3')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[5]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_4')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[6]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_5')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[7]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_6')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[8]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_7')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[9]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_8')))
    tf.summary.image('%s/generated_%s'%(labels[0],labels[10]),
            utils.batch_convert2int(self.G(self.reader_set[0],deconder_name='twin_deconder_9')))
    tf.summary.image('%s/generated_%s'%(labels[0],'mask'),utils.batch_convert2int(self.G.FCN_mask))
    self.raw_image_generated_images0 = [ self.G(self.reader_set[0],deconder_name='twin_deconder_%d'%i) for i in xrange(10)]
    self.raw_image_generated_images0.insert(0,self.reader_set[0])
 #   tf.summary.image('X/reconstruction_from_Y', utils.batch_convert2int(self.F(G_x)))
 #   tf.summary.image('X/reconstruction_from_former_features_to_Y', utils.batch_convert2int(F_x))
 #   tf.summary.image('X/reconstruction_from_Z', utils.batch_convert2int(self.F(G_from_x_to_z, enconder_name ='twin_enconder_1')))
 #   tf.summary.image('X/reconstruction_from_former_features_to_Z', utils.batch_convert2int(F_x_z))
 #  # tf.summary.image('X/generated_Y', utils.batch_convert2int(self.G(x)))
 #  # tf.summary.image('X/reconstruction_from_Y', utils.batch_convert2int(self.F(self.G(x))))
 #  # tf.summary.image('X/reconstruction_from_former_features_to_Y', utils.batch_convert2int(F_x))
 #   ################################################################################################################
 #
 #   ################################################################################################################
    F_y,F_features = self.F(self.reader_set[1],output_media_features = True)
    tf.summary.image('%s/generated_%s'%(labels[1],labels[0]),
            utils.batch_convert2int(F_y))

    tf.summary.image('%s/generated_%s'%(labels[1],labels[2]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_1') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[3]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_2') # F_y actually doesnot work
                    ))

    self.raw_image_generated_images1 = [F_y] + [self.G(F_y,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)]
    self.raw_image_generated_images1[1] = self.reader_set[1]
    tf.summary.image('%s/generated_%s'%(labels[1],labels[4]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_3') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[5]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_4') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[6]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_5') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[7]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_6') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[8]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_7') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[9]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_8') # F_y actually doesnot work
                    ))
    tf.summary.image('%s/generated_%s'%(labels[1],labels[10]), utils.batch_convert2int(
        self.G(F_y,use_media_features_from_former_network =F_features,
            use_media_features = True,deconder_name='twin_deconder_9') # F_y actually doesnot work
                    ))

 #   tf.summary.image('Y/generated_X', utils.batch_convert2int(F_y))
 #   tf.summary.image('Y/reconstruction_from_X', utils.batch_convert2int(self.G(F_y)))
 #   tf.summary.image('Y/reconstruction_from_former_features', utils.batch_convert2int(G_y))
 #   ################################################################################################################
 #
    F_z,F_z_features = self.F(self.reader_set[2],output_media_features = True,enconder_name ='twin_enconder_1')
    tf.summary.image('%s/generated_%s'%(labels[2],labels[0]), utils.batch_convert2int(F_z))


 #   G_x_from_z = self.G(F_z,use_media_features_from_former_network =F_z_features,use_media_features = True,deconder_name='twin_deconder_1') # F_z actually doesnot work
    tf.summary.image('%s/generated_%s'%(labels[2],labels[1]), utils.batch_convert2int(
        self.G(F_z,use_media_features_from_former_network =F_z_features,
            use_media_features = True,deconder_name='twin_deconder_0') # F_z actually doesnot work
        ))

    tf.summary.image('%s/generated_%s'%(labels[2],labels[3]), utils.batch_convert2int(
        self.G(F_z,use_media_features_from_former_network =F_z_features,
            use_media_features = True,deconder_name='twin_deconder_2') # F_z actually doesnot work
        ))
    self.raw_image_generated_images2 = [F_z] + [self.G(F_z,use_media_features_from_former_network =F_z_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)]
    self.raw_image_generated_images2[2] = self.reader_set[2]
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[4]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_3') # F_z actually doesnot work
    #    ))
    #
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[5]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_4') # F_z actually doesnot work
    #    ))
    #
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[6]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_5') # F_z actually doesnot work
    #    ))
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[7]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_6') # F_z actually doesnot work
    #    ))
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[8]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_7') # F_z actually doesnot work
    #    ))
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[9]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_8') # F_z actually doesnot work
    #    ))
    #tf.summary.image('%s/generated_%s'%(labels[2],labels[10]), utils.batch_convert2int(
    #    self.G(F_z,use_media_features_from_former_network =F_z_features,
    #        use_media_features = True,deconder_name='twin_deconder_9') # F_z actually doesnot work
    #    ))
 #   tf.summary.image('Z/generated_X', utils.batch_convert2int(F_z))
 #   tf.summary.image('Z/reconstruction_from_X', utils.batch_convert2int(self.G(F_z,deconder_name='twin_deconder_1')))
 ##   tf.summary.image('Z/reconstruction_from_former_features', utils.batch_convert2int(G_x_from_z))
  #  tf.summary.image('train_2/generated_y', utils.batch_convert2int(G_y_from_z))
  #  tf.summary.image('train_2/generated_y_then_x', utils.batch_convert2int(self.F(G_y_from_z)))
 ##
  #  tf.summary.image('train_2/generated_W', utils.batch_convert2int(G_w_2))
 ##   #return G_loss+G_loss_Z, D_Y_loss+ D_Z_loss, F_loss+F_loss_Z, D_X_loss+D_X_loss_from_Z, fake_y, fake_x,fake_z
    F_h,F_h_features = self.F(self.reader_set[3],output_media_features = True,enconder_name ='twin_enconder_2')
    tf.summary.image('%s/generated_%s'%(labels[3],labels[0]), utils.batch_convert2int(F_h))


 #   G_x_from_z = self.G(F_z,use_media_features_from_former_network =F_z_features,use_media_features = True,deconder_name='twin_deconder_1') # F_z actually doesnot work
    tf.summary.image('%s/generated_%s'%(labels[3],labels[1]), utils.batch_convert2int(
        self.G(F_h,use_media_features_from_former_network =F_h_features,
            use_media_features = True,deconder_name='twin_deconder_0') # F_z actually doesnot work
        ))

    tf.summary.image('%s/generated_%s'%(labels[3],labels[2]), utils.batch_convert2int(
        self.G(F_h,use_media_features_from_former_network =F_h_features,
            use_media_features = True,deconder_name='twin_deconder_1') # F_z actually doesnot work
        ))
    self.raw_image_generated_images3 = [F_h] + [self.G(F_h,use_media_features_from_former_network =F_h_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)]
    self.raw_image_generated_images3[3] = self.reader_set[3]

    F_,F_features = self.F(self.reader_set[4],output_media_features = True,enconder_name ='twin_enconder_3')
    self.raw_image_generated_images4 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images4[4] = self.reader_set[4]
 
    F_,F_features = self.F(self.reader_set[5],output_media_features = True,enconder_name ='twin_enconder_4')
    self.raw_image_generated_images5 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images5[5] = self.reader_set[5]
 
    F_,F_features = self.F(self.reader_set[6],output_media_features = True,enconder_name ='twin_enconder_5')
    self.raw_image_generated_images6 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images6[6] = self.reader_set[6]
 
    F_,F_features = self.F(self.reader_set[7],output_media_features = True,enconder_name ='twin_enconder_6')
    self.raw_image_generated_images7 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images7[7] = self.reader_set[7]
 
    F_,F_features = self.F(self.reader_set[8],output_media_features = True,enconder_name ='twin_enconder_7')
    self.raw_image_generated_images8 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images8[8] = self.reader_set[8]
 
    F_,F_features = self.F(self.reader_set[9],output_media_features = True,enconder_name ='twin_enconder_8')
    self.raw_image_generated_images9 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images9[9] = self.reader_set[9]
    F_,F_features = self.F(self.reader_set[10],output_media_features = True,enconder_name ='twin_enconder_9')
    self.raw_image_generated_images10 = [F_] +  [self.G(F_,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name='twin_deconder_%d'%i) for i in xrange(10)] 
    self.raw_image_generated_images10[10] = self.reader_set[10]
 
    self.raw_image_generated_images = self.raw_image_generated_images0+self.raw_image_generated_images1+self.raw_image_generated_images2+self.raw_image_generated_images3+self.raw_image_generated_images4+ \
                                        self.raw_image_generated_images5+self.raw_image_generated_images6+self.raw_image_generated_images7+self.raw_image_generated_images8+self.raw_image_generated_images9+ \
        				self.raw_image_generated_images10
 # 
    return self.G_y_set, self.D_y_set, self.F_x_set, self.D_x_set
  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    # building the whole network, parameters in encoders for domains excepte for anchor are not used  as wellas parameters in decoder for anchor. To accelerate the operation of the program, unwanted parameters are not updated.
    G_not_use_var = ['twin_enconder_%d'%(i+1) for i in xrange(self.number_domain -2)]
    G_train_var = [v for v in self.G.variables if v.name[2:17] not in G_not_use_var ]
    G_optimizer = make_optimizer(G_loss, G_train_var, name='Adam_G')
    #G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_var = self.D_set[1].variables
    for i in xrange(self.number_domain -2):
	D_Y_var +=self.D_set[i+2].variables 
    D_Y_optimizer = make_optimizer(D_Y_loss, D_Y_var, name='Adam_D_Y')

    F_not_use_var = ['twin_deconder_%d'%(i+1) for i in xrange(self.number_domain -2)]
    F_train_var = [v for v in self.F.variables if v.name[2:17] not in F_not_use_var ]
    F_optimizer =  make_optimizer(F_loss, F_train_var, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_set[0].variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
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
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def cycle_consistency_loss(self, special_domain = 0):
    """ cycle consistency loss (L1 norm)
        enconder_name,deconder_name: to control encoder and deconder share or not
    """
    G,F = self.G, self.F 
    x = self.reader_set[0]
    y = self.reader_set[special_domain]
    enconder_name ='twin_enconder_%d'%(special_domain-1)
    deconder_name='twin_deconder_%d'%(special_domain-1)
    #### doesnot make sense it is optimal way
    G.reuse,F.reuse = False,False,
    _ =F(G(x,enconder_name =enconder_name,deconder_name=deconder_name),enconder_name =enconder_name,deconder_name=deconder_name)

    G_x,G_features = G(x,output_media_features = True,deconder_name=deconder_name)
    F_G_x = F(G_x,enconder_name =enconder_name)
    F_x = F(G_x,use_media_features_from_former_network =G_features,use_media_features = True,deconder_name=deconder_name) # G_x actually doesnot work
    forward_loss = tf.reduce_mean(tf.abs(F_G_x-x)) + tf.reduce_mean(tf.abs(F_x-x))

####cause just x is shared, y and z donot share
    F_y,F_features = F(y,output_media_features = True,enconder_name =enconder_name)
    G_F_y = G(F_y,deconder_name=deconder_name)
    G_y = G(F_y,use_media_features_from_former_network =F_features,use_media_features = True,deconder_name=deconder_name) # F_x actually doesnot work
    backward_loss = tf.reduce_mean(tf.abs(G_F_y-y)) + tf.reduce_mean(tf.abs(G_y-y))

    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss
  def gan_cycle_loss(self, special_domain =0):
	    # X -> Z
	    x = self.reader_set[0]
	    z = self.reader_set[special_domain]
	    cycle_loss_z = self.cycle_loss_set[special_domain-1]
	    fake_x_pool = self.fake_set[0]
	    fake_z_pool = self.fake_set[special_domain]
	    D_X = self.D_set[0]
	    D_X_Z = self.D_set[special_domain]
	    ################################################################################################
	    enconder_name ='twin_enconder_%d'%(special_domain-1)
	    deconder_name='twin_deconder_%d'%(special_domain-1)
	    fake_z = self.G(x,deconder_name=deconder_name)                                             #
	    G_gan_loss = self.generator_loss(D_X_Z, fake_z, use_lsgan=self.use_lsgan)                 # 
	    G_loss_Z =  G_gan_loss + cycle_loss_z                                                       #
	    D_Z_loss = self.discriminator_loss(D_X_Z, z, fake_z_pool, use_lsgan=self.use_lsgan)       #
	    ################################################################################################
	    # Z -> X
	    ################################################################################################
	    fake_x_from_z = self.F(z,enconder_name =enconder_name)
	    F_gan_loss_from_z = self.generator_loss(D_X, fake_x_from_z, use_lsgan=self.use_lsgan)
	    F_loss_Z = F_gan_loss_from_z + cycle_loss_z
	    D_X_loss = self.discriminator_loss(D_X, x, fake_x_pool, use_lsgan=self.use_lsgan)
	    return (G_loss_Z,D_Z_loss,F_loss_Z,D_X_loss,fake_z,fake_x_from_z) 

