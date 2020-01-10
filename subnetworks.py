# -*- coding: utf-8 -*-

import tensorflow as tf
from components import *
import tensorflow.contrib.slim as slim
from config import cfg

def unet(input_tensor):
	'''
	conv1 = slim.convolution2d(input_tensor, 6, [3,3], stride=1, padding='SAME', scope='conv1_1')
	conv1 = slim.batch_norm(conv1, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv1_1/batch_norm')
	conv1 = slim.convolution2d(conv1, 6, [3,3], stride=1, padding='SAME', scope='conv1_2')
	conv1 = slim.batch_norm(conv1, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv1_2/batch_norm')
	conv2 = slim.max_pool2d(conv1, [2,2], stride=2, scope='pool1')
	conv2 = slim.convolution2d(conv2, 9, [3,3], stride=1, padding='SAME', scope='conv2_1')
	conv2 = slim.batch_norm(conv2, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv2_1/batch_norm')
	conv2 = slim.convolution2d(conv2, 9, [3,3], stride=1, padding='SAME', scope='conv2_2')
	conv2 = slim.batch_norm(conv2, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv2_2/batch_norm')
	conv3 = slim.max_pool2d(conv2, [2,2], stride=2, scope='pool2')	
	conv3 = slim.convolution2d(conv3, 12, [3,3], stride=1, padding='SAME', scope='conv3_1')
	conv3 = slim.batch_norm(conv3, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv3_1/batch_norm')
	conv3 = slim.convolution2d(conv3, 12, [3,3], stride=1, padding='SAME', scope='conv3_2')
	conv3 = slim.batch_norm(conv3, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv3_2/batch_norm')
	conv4 = slim.max_pool2d(conv3, [2,2], stride=2, scope='pool3')
	conv4 = slim.convolution2d(conv4, 15, [3,3], stride=1, padding='SAME', scope='conv4_1')
	conv4 = slim.batch_norm(conv4, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv4_1/batch_norm')
	conv4 = slim.convolution2d(conv4, 15, [3,3], stride=1, padding='SAME', scope='conv4_2')
	conv4 = slim.batch_norm(conv4, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv4_2/batch_norm')
	conv5 = slim.max_pool2d(conv4, [2,2], stride=2, scope='pool4')
	conv5 = slim.convolution2d(conv5, 23, [3,3], stride=1, padding='SAME', scope='conv5_1')
	conv5 = slim.batch_norm(conv5, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv5_1/batch_norm')
	conv5 = slim.convolution2d(conv5, 23, [3,3], stride=1, padding='SAME', scope='conv5_2')
	conv5 = slim.batch_norm(conv5, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv5_2/batch_norm')
	conv6 = slim.max_pool2d(conv5, [2,2], stride=2, scope='pool5')
	conv6 = slim.convolution2d(conv6, 31, [3,3], stride=1, padding='SAME', scope='conv6_1')
	conv6 = slim.batch_norm(conv6, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv6_1/batch_norm')
	conv6 = slim.convolution2d(conv6, 31, [3,3], stride=1, padding='SAME', scope='conv6_2')
	conv6 = slim.batch_norm(conv6, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv6_2/batch_norm')
	conv7 = tf.image.resize_images(conv6, [conv5.shape[1], conv5.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv7 = tf.concat([conv7, conv5], axis=-1, name='concat6')
	conv7 = slim.convolution2d(conv7, 23, [3,3], stride=1, padding='SAME', scope='conv7_1')
	conv7 = slim.batch_norm(conv7, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv7_1/batch_norm')
	conv7 = slim.convolution2d(conv7, 23, [3,3], stride=1, padding='SAME', scope='conv7_2')
	conv7 = slim.batch_norm(conv7, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv7_2/batch_norm')
	conv8 = tf.image.resize_images(conv7, [conv4.shape[1], conv4.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv8 = tf.concat([conv8, conv4], axis=-1, name='concat7')
	conv8 = slim.convolution2d(conv8, 15, [3,3], stride=1, padding='SAME', scope='conv8_1')
	conv8 = slim.batch_norm(conv8, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv8_1/batch_norm')
	conv8 = slim.convolution2d(conv8, 15, [3,3], stride=1, padding='SAME', scope='conv8_2')
	conv8 = slim.batch_norm(conv8, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv8_2/batch_norm')
	conv9 = tf.image.resize_images(conv8, [conv3.shape[1], conv3.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv9 = tf.concat([conv9, conv3], axis=-1, name='concat8')
	conv9 = slim.convolution2d(conv9, 12, [3,3], stride=1, padding='SAME', scope='conv9_1')
	conv9 = slim.batch_norm(conv9, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv9_1/batch_norm')
	conv9 = slim.convolution2d(conv9, 12, [3,3], stride=1, padding='SAME', scope='conv9_2')
	conv9 = slim.batch_norm(conv9, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv9_2/batch_norm')
	conv10 = tf.image.resize_images(conv9, [conv2.shape[1], conv2.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv10 = tf.concat([conv10, conv2], axis=-1, name='concat9')
	conv10 = slim.convolution2d(conv10, 9, [3,3], stride=1, padding='SAME', scope='conv10_1')
	conv10 = slim.batch_norm(conv10, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv10_1/batch_norm')
	conv10 = slim.convolution2d(conv10, 9, [3,3], stride=1, padding='SAME', scope='conv10_2')
	conv10 = slim.batch_norm(conv10, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv10_2/batch_norm')
	conv11 = tf.image.resize_images(conv10, [conv1.shape[1], conv1.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv11 = tf.concat([conv11, conv1], axis=-1, name='concat10')
	conv11 = slim.convolution2d(conv11, 6, [3,3], stride=1, padding='SAME', scope='conv11_1')
	conv11 = slim.batch_norm(conv11, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv11_1/batch_norm')
	conv11 = slim.convolution2d(conv11, 6, [3,3], stride=1, padding='SAME', scope='conv11_2')
	conv11 = slim.batch_norm(conv11, is_training=cfg.is_training,
					activation_fn=tf.nn.relu,
					fused=True, scope='conv11_2/batch_norm')
	conv11 = slim.convolution2d(conv11, cfg.num_class, [3,3], stride=1, padding='SAME', scope='conv_output')
	'''
	conv1 = tf.layers.conv2d(input_tensor, filters=6, kernel_size=[3,3], strides=1, padding='SAME', name='conv1_1')
	conv1 = tf.layers.batch_normalization(conv1, training=cfg.is_training, name='conv1_1/batch_norm')
	conv1 = tf.nn.relu(conv1, name='conv1_1/relu')
	conv1 = tf.layers.conv2d(conv1, 6, [3,3], strides=1, padding='SAME', name='conv1_2')
	conv1 = tf.layers.batch_normalization(conv1, training=cfg.is_training, name='conv1_2/batch_norm')
	conv1 = tf.nn.relu(conv1, name='conv1_2/relu')
	conv2 = tf.layers.max_pooling2d(conv1, [2,2], strides=2, name='pool1')
	conv2 = tf.layers.conv2d(conv2, 9, [3,3], strides=1, padding='SAME', name='conv2_1')
	conv2 = tf.layers.batch_normalization(conv2, training=cfg.is_training, name='conv2_1/batch_norm')
	conv2 = tf.nn.relu(conv2, name='conv2_1/relu')
	conv2 = tf.layers.conv2d(conv2, 9, [3,3], strides=1, padding='SAME', name='conv2_2')
	conv2 = tf.layers.batch_normalization(conv2, training=cfg.is_training, name='conv2_2/batch_norm')
	conv2 = tf.nn.relu(conv2, name='conv2_2/relu')
	conv3 = tf.layers.max_pooling2d(conv2, [2,2], strides=2, name='pool2')	
	conv3 = tf.layers.conv2d(conv3, 12, [3,3], strides=1, padding='SAME', name='conv3_1')
	conv3 = tf.layers.batch_normalization(conv3, training=cfg.is_training, name='conv3_1/batch_norm')
	conv3 = tf.nn.relu(conv3, name='conv3_1/relu')
	conv3 = tf.layers.conv2d(conv3, 12, [3,3], strides=1, padding='SAME', name='conv3_2')
	conv3 = tf.layers.batch_normalization(conv3, training=cfg.is_training, name='conv3_2/batch_norm')
	conv3 = tf.nn.relu(conv3, name='conv3_2/relu')
	conv4 = tf.layers.max_pooling2d(conv3, [2,2], strides=2, name='pool3')
	conv4 = tf.layers.conv2d(conv4, 15, [3,3], strides=1, padding='SAME', name='conv4_1')
	conv4 = tf.layers.batch_normalization(conv4, training=cfg.is_training, name='conv4_1/batch_norm')
	conv4 = tf.nn.relu(conv4, name='conv4_1/relu')
	conv4 = tf.layers.conv2d(conv4, 15, [3,3], strides=1, padding='SAME', name='conv4_2')
	conv4 = tf.layers.batch_normalization(conv4, training=cfg.is_training, name='conv4_2/batch_norm')
	conv4 = tf.nn.relu(conv3, name='conv4_2/relu')
	conv5 = tf.layers.max_pooling2d(conv4, [2,2], strides=2, name='pool4')
	conv5 = tf.layers.conv2d(conv5, 23, [3,3], strides=1, padding='SAME', name='conv5_1')
	conv5 = tf.layers.batch_normalization(conv5, training=cfg.is_training, name='conv5_1/batch_norm')
	conv5 = tf.nn.relu(conv5, name='conv5_1/relu')
	conv5 = tf.layers.conv2d(conv5, 23, [3,3], strides=1, padding='SAME', name='conv5_2')
	conv5 = tf.layers.batch_normalization(conv5, training=cfg.is_training, name='conv5_2/batch_norm')
	conv5 = tf.nn.relu(conv5, name='conv5_2/relu')
	conv6 = tf.layers.max_pooling2d(conv5, [2,2], strides=2, name='pool5')
	conv6 = tf.layers.conv2d(conv6, 31, [3,3], strides=1, padding='SAME', name='conv6_1')
	conv6 = tf.layers.batch_normalization(conv6, training=cfg.is_training, name='conv6_1/batch_norm')
	conv6 = tf.nn.relu(conv6, name='conv6_1/relu')
	conv6 = tf.layers.conv2d(conv6, 31, [3,3], strides=1, padding='SAME', name='conv6_2')
	conv6 = tf.layers.batch_normalization(conv6, training=cfg.is_training, name='conv6_2/batch_norm')
	conv6 = tf.nn.relu(conv6, name='conv6_2/relu')
	conv7 = tf.image.resize_images(conv6, [conv5.shape[1], conv5.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv7 = tf.concat([conv7, conv5], axis=-1, name='concat6')
	conv7 = tf.layers.conv2d(conv7, 23, [3,3], strides=1, padding='SAME', name='conv7_1')
	conv7 = tf.layers.batch_normalization(conv7, training=cfg.is_training, name='conv7_1/batch_norm')
	conv7 = tf.nn.relu(conv7, name='conv7_1/relu')
	conv7 = tf.layers.conv2d(conv7, 23, [3,3], strides=1, padding='SAME', name='conv7_2')
	conv7 = tf.layers.batch_normalization(conv7, training=cfg.is_training, name='conv7_2/batch_norm')
	conv7 = tf.nn.relu(conv7, name='conv7_2/relu')
	conv8 = tf.image.resize_images(conv7, [conv4.shape[1], conv4.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv8 = tf.concat([conv8, conv4], axis=-1, name='concat7')
	conv8 = tf.layers.conv2d(conv8, 15, [3,3], strides=1, padding='SAME', name='conv8_1')
	conv8 = tf.layers.batch_normalization(conv8, training=cfg.is_training, name='conv8_1/batch_norm')
	conv8 = tf.nn.relu(conv8, name='conv8_1/relu')
	conv8 = tf.layers.conv2d(conv8, 15, [3,3], strides=1, padding='SAME', name='conv8_2')
	conv8 = tf.layers.batch_normalization(conv8, training=cfg.is_training, name='conv8_2/batch_norm')
	conv8 = tf.nn.relu(conv8, name='conv8_2/relu')
	conv9 = tf.image.resize_images(conv8, [conv3.shape[1], conv3.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv9 = tf.concat([conv9, conv3], axis=-1, name='concat8')
	conv9 = tf.layers.conv2d(conv9, 12, [3,3], strides=1, padding='SAME', name='conv9_1')
	conv9 = tf.layers.batch_normalization(conv9, training=cfg.is_training, name='conv9_1/batch_norm')
	conv9 = tf.nn.relu(conv9, name='conv9_1/relu')
	conv9 = tf.layers.conv2d(conv9, 12, [3,3], strides=1, padding='SAME', name='conv9_2')
	conv9 = tf.layers.batch_normalization(conv9, training=cfg.is_training, name='conv9_2/batch_norm')
	conv9 = tf.nn.relu(conv9, name='conv9_2/relu')
	conv10 = tf.image.resize_images(conv9, [conv2.shape[1], conv2.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv10 = tf.concat([conv10, conv2], axis=-1, name='concat9')
	conv10 = tf.layers.conv2d(conv10, 9, [3,3], strides=1, padding='SAME', name='conv10_1')
	conv10 = tf.layers.batch_normalization(conv10, training=cfg.is_training, name='conv10_1/batch_norm')
	conv10 = tf.nn.relu(conv10, name='conv10_1/relu')
	conv10 = tf.layers.conv2d(conv10, 9, [3,3], strides=1, padding='SAME', name='conv10_2')
	conv10 = tf.layers.batch_normalization(conv10, training=cfg.is_training, name='conv10_2/batch_norm')
	conv10 = tf.nn.relu(conv10, name='conv10_2/relu')
	conv11 = tf.image.resize_images(conv10, [conv1.shape[1], conv1.shape[2]],
									method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv11 = tf.concat([conv11, conv1], axis=-1, name='concat10')
	conv11 = tf.layers.conv2d(conv11, 6, [3,3], strides=1, padding='SAME', name='conv11_1')
	conv11 = tf.layers.batch_normalization(conv11, training=cfg.is_training, name='conv11_1/batch_norm')
	conv11 = tf.nn.relu(conv11, name='conv11_1/relu')
	conv11 = tf.layers.conv2d(conv11, 6, [3,3], strides=1, padding='SAME', name='conv11_2')
	conv11 = tf.layers.batch_normalization(conv11, training=cfg.is_training, name='conv11_2/batch_norm')
	conv11 = tf.nn.relu(conv11, name='conv11_2/relu')
	conv11 = tf.layers.conv2d(conv11, cfg.num_class, [3,3], strides=1, padding='SAME', name='conv_output')

	return conv11
