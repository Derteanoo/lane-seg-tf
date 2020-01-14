# -*- coding: utf-8 -*-

import tensorflow as tf
from components import *
import tensorflow.contrib.slim as slim
from config import cfg

def unet(input_tensor):
	batch_norm_params={'decay':0.997,
					   'epsilon':0.001,
					   'is_training': cfg.is_training
					   #'updatas_collections':tf.GraphKeys.UPDATE_OPS,
					   #'variables_collections':{
						#   'beta':None,
						#   'gamma':None,
						#   'moving_mean':'moving_vars',
						#   'moving_variance':'moviong_vars'
					   #}
					   }

	conv1 = slim.convolution2d(input_tensor, 6, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv1_1')
	conv1 = slim.convolution2d(conv1, 6, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv1_2')
	conv2 = slim.max_pool2d(conv1, [2,2], stride=2, scope='pool1')
	conv2 = slim.convolution2d(conv2, 9, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv2_1')
	conv2 = slim.convolution2d(conv2, 9, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv2_2')
	conv3 = slim.max_pool2d(conv2, [2,2], stride=2, scope='pool2')	
	conv3 = slim.convolution2d(conv3, 12, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv3_1')
	conv3 = slim.convolution2d(conv3, 12, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv3_2')
	conv4 = slim.max_pool2d(conv3, [2,2], stride=2, scope='pool3')
	conv4 = slim.convolution2d(conv4, 15, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv4_1')
	conv4 = slim.convolution2d(conv4, 15, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv4_2')
	conv5 = slim.max_pool2d(conv4, [2,2], stride=2, scope='pool4')
	conv5 = slim.convolution2d(conv5, 23, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv5_1')
	conv5 = slim.convolution2d(conv5, 23, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv5_2')
	conv6 = slim.max_pool2d(conv5, [2,2], stride=2, scope='pool5')
	conv6 = slim.convolution2d(conv6, 31, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv6_1')
	conv6 = slim.convolution2d(conv6, 31, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv6_2')
	conv7 = tf.image.resize_images(conv6, [conv5.shape[1], conv5.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv7 = tf.concat([conv7, conv5], axis=-1, name='concat6')
	conv7 = slim.convolution2d(conv7, 23, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv7_1')
	conv7 = slim.convolution2d(conv7, 23, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv7_2')
	conv8 = tf.image.resize_images(conv7, [conv4.shape[1], conv4.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv8 = tf.concat([conv8, conv4], axis=-1, name='concat7')
	conv8 = slim.convolution2d(conv8, 15, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv8_1')
	conv8 = slim.convolution2d(conv8, 15, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv8_2')
	conv9 = tf.image.resize_images(conv8, [conv3.shape[1], conv3.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv9 = tf.concat([conv9, conv3], axis=-1, name='concat8')
	conv9 = slim.convolution2d(conv9, 12, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv9_1')
	conv9 = slim.convolution2d(conv9, 12, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv9_2')
	conv10 = tf.image.resize_images(conv9, [conv2.shape[1], conv2.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv10 = tf.concat([conv10, conv2], axis=-1, name='concat9')
	conv10 = slim.convolution2d(conv10, 9, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv10_1')
	conv10 = slim.convolution2d(conv10, 9, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv10_2')
	conv11 = tf.image.resize_images(conv10, [conv1.shape[1], conv1.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv11 = tf.concat([conv11, conv1], axis=-1, name='concat10')
	conv11 = slim.convolution2d(conv11, 6, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv11_1')
	conv11 = slim.convolution2d(conv11, 6, [3,3], stride=1, padding='SAME', 
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv11_2')
	conv11 = slim.convolution2d(conv11, cfg.num_class, [3,3], stride=1, padding='SAME', activation_fn=None, scope='conv_output')

	return conv11
