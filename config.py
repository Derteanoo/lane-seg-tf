# -*- coding: utf-8 -*-

import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For focal loss
flags.DEFINE_float('alpha', 0.25, 'coefficient for focal loss')
flags.DEFINE_float('gamma', 2.0, 'factor for focal loss')
flags.DEFINE_float('fl_weight', 0.1, 'regularization coefficient for focal loss')

# For training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 400, 'epoch')

flags.DEFINE_float('init_lr', 0.005, 'initial learning rate')
flags.DEFINE_float('beta1', 0.5, 'exponential decay rate of the first moment of adam learning rate')
flags.DEFINE_float('beta2', 0.999, 'exponential decay rate of the second moment of adam learning rate')
flags.DEFINE_float('power', 0.9, 'decay factor of learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum factor')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.0001, 'regularization coefficient for W and b')

flags.DEFINE_integer('horizon', 360, 'the top of crop region')
flags.DEFINE_integer('width', 128, 'the width of segmentation image')
flags.DEFINE_integer('height', 128, 'the height of segmentation image')
flags.DEFINE_integer('num_class', 5, 'the number of class')

############################
#   environment setting    #
############################
flags.DEFINE_boolean('is_training', False, 'train or test phase')
flags.DEFINE_string('images', 'data', 'The root directory of dataset')
flags.DEFINE_string('logdir', 'logs', 'logs directory')
flags.DEFINE_string('log', 'trainval.log', 'log file')
flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 50, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 5, 'the frequency of saving model(step)')
flags.DEFINE_string('models', 'models', 'path for saving models')
flags.DEFINE_string('test_outputs', 'test-outputs', 'path for saving test results')
flags.DEFINE_boolean('is_quantize', False, 'quantize training or not')
flags.DEFINE_boolean('is_scratch', True, 'training from scratch or not')
flags.DEFINE_boolean('is_write_pb', False, 'save pb or not when testing')

cfg = tf.app.flags.FLAGS

# tf.logging.set_verbosity(tf.logging.INFO)
