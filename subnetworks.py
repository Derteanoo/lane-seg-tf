# -*- coding: utf-8 -*-

import tensorflow as tf
from components import *
import tensorflow.contrib.slim as slim
from config import cfg

def unet(input_tensor):
	width_mult = 1.

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

	conv1 = slim.convolution2d(input_tensor, 32 * width_mult, [3,3], stride=2, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv1_1')
	
	conv1 = slim.separable_conv2d(conv1, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv1_2')#1

	conv1 = slim.convolution2d(conv1, 16 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv1_3')#1

	conv2 = slim.convolution2d(conv1, 96 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv2_1')
		
	conv2 = slim.separable_conv2d(conv2, num_outputs=None, kernel_size=[3,3], stride=2, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv2_2')

	conv2 = slim.convolution2d(conv2, 24 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv2_3')

	conv3 = slim.convolution2d(conv2, 144 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv3_1')
		
	conv3 = slim.separable_conv2d(conv3, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv3_2')

	conv3 = slim.convolution2d(conv3, 24 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv3_3')

	add3 = tf.add(conv2, conv3)#2

	conv4 = slim.convolution2d(add3, 144 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv4_1')
		
	conv4 = slim.separable_conv2d(conv4, num_outputs=None, kernel_size=[3,3], stride=2, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv4_2')	

	conv4 = slim.convolution2d(conv4, 32 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv4_3')		 

	conv5 = slim.convolution2d(conv4, 192 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv5_1')
		
	conv5 = slim.separable_conv2d(conv5, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv5_2')	

	conv5 = slim.convolution2d(conv5, 32 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv5_3')	

	add5 = tf.add(conv4, conv5)#3

	conv6 = slim.convolution2d(add5, 192 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv6_1')
		
	conv6 = slim.separable_conv2d(conv6, num_outputs=None, kernel_size=[3,3], stride=2, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv6_2')	

	conv6 = slim.convolution2d(conv6, 64 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv6_3')	

	conv7 = slim.convolution2d(conv6, 384 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv7_1')
		
	conv7 = slim.separable_conv2d(conv7, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv7_2')	

	conv7 = slim.convolution2d(conv7, 64 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv7_3')	

	add8 = tf.add(conv6, conv7)

	conv9 = slim.convolution2d(add8, 384 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv9_1')
	
	conv9 = slim.separable_conv2d(conv9, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv9_2')	

	conv9 = slim.convolution2d(conv9, 64 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv9_3')	

	add9 = tf.add(add8, conv9)

	conv10 = slim.convolution2d(add9, 384 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv10_1')
		
	conv10 = slim.separable_conv2d(conv10, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv10_2')	

	conv10 = slim.convolution2d(conv10, 64 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv10_3')	

	add10 = tf.add(add9, conv10)

	conv11 = slim.convolution2d(add10, 384 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv11_1')
		
	conv11 = slim.separable_conv2d(conv11, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv11_2')	

	conv11 = slim.convolution2d(conv11, 96 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv11_3')	

	conv12 = slim.convolution2d(conv11, 576 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv12_1')
		
	conv12 = slim.separable_conv2d(conv12, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv12_2')	

	conv12 = slim.convolution2d(conv12, 96 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv12_3')	
	
	add12 = tf.add(conv11, conv12)

	conv13 = slim.convolution2d(add12, 576 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv13_1')
		
	conv13 = slim.separable_conv2d(conv13, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv13_2')	

	conv13 = slim.convolution2d(conv13, 96 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv13_3')	
	
	add13 = tf.add(add12, conv13)#4

	conv14 = slim.convolution2d(add13, 576 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv14_1')
		
	conv14 = slim.separable_conv2d(conv14, num_outputs=None, kernel_size=[3,3], stride=2, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv14_2')	

	conv14 = slim.convolution2d(conv14, 160 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv14_3')	

	conv15 = slim.convolution2d(conv14, 960 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv15_1')
		
	conv15 = slim.separable_conv2d(conv15, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv15_2')	

	conv15 = slim.convolution2d(conv15, 160 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv15_3')	

	add15 = tf.add(conv14, conv15)

	conv16 = slim.convolution2d(add15, 960 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv16_1')
		
	conv16 = slim.separable_conv2d(conv16, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv16_2')	

	conv16 = slim.convolution2d(conv16, 160 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv16_3')	
	
	add16 = tf.add(conv15, conv15)

	conv17 = slim.convolution2d(add16, 960 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv17_1')
		
	conv17 = slim.separable_conv2d(conv17, num_outputs=None, kernel_size=[3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv17_2')	

	conv17 = slim.convolution2d(conv17, 320 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.identity, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv17_3')	

	upsample17 = tf.image.resize_images(conv17, [conv13.shape[1], conv13.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)	

	conv18 = slim.convolution2d(upsample17, 96 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv18_1')

	add18 = tf.add(add13, conv18)

	conv19 = slim.convolution2d(add18, 96 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv19_1')

	conv19 = slim.convolution2d(conv19, 96 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv19_2')

	upsample19 = tf.image.resize_images(conv19, [conv5.shape[1], conv5.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)							 

	conv20 = slim.convolution2d(upsample19, 32 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv20_1')

	add20 = tf.add(add5, conv20)

	conv21 = slim.convolution2d(add20, 32 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv21_1')

	conv21 = slim.convolution2d(conv21, 32 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv21_2')

	upsample21 = tf.image.resize_images(conv21, [conv3.shape[1], conv3.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)							 

	conv22 = slim.convolution2d(upsample21, 24 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv22_1')

	add22 = tf.add(add3, conv22)

	conv23 = slim.convolution2d(add22, 24 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv23_1')

	conv23 = slim.convolution2d(conv23, 24 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv23_2')

	upsample23 = tf.image.resize_images(conv23, [conv1.shape[1], conv1.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)							 

	conv24 = slim.convolution2d(upsample23, 16 * width_mult, [1,1], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv24_1')

	add24 = tf.add(conv1, conv24)

	conv25 = slim.convolution2d(add24, 16 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv25_1')

	conv25 = slim.convolution2d(conv25, 16 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv25_2')

	upsample25 = tf.image.resize_images(conv25, [input_tensor.shape[1], input_tensor.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)	

	conv26 = slim.convolution2d(upsample25, 16 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv26_1')

	conv26 = slim.convolution2d(conv26, 16 * width_mult, [3,3], stride=1, padding='SAME',
	                             activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
								 normalizer_params=batch_norm_params, scope='conv26_2')

	conv27 = slim.convolution2d(conv26,  cfg.num_class, [3,3], stride=1, padding='SAME', scope='conv_27')

	conv28 = tf.identity(conv27, name='conv_output')


	return conv28
