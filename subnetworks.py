# -*- coding: utf-8 -*-

import tensorflow as tf
from components import *
import tensorflow.contrib.slim as slim
from config import cfg

def unet(input_tensor):
	cfg.is_training=True
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
									method=tf.image.ResizeMethod.BILINEAR)
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
									method=tf.image.ResizeMethod.BILINEAR)
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
									method=tf.image.ResizeMethod.BILINEAR)
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
									method=tf.image.ResizeMethod.BILINEAR)
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
									method=tf.image.ResizeMethod.BILINEAR)
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
	cfg.is_training=False
	return conv11
