# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from subnetworks import *
from losses import *
from config import cfg
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

def cal_iou(y_true, y_pred):
	'''
	overlap_map = y_true * y_pred
	
	try:
		
		gt_count = tf.reduce_sum(tf.cast(y_true, tf.float32), [1, 2], keepdims=True)
		pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32), [1, 2], keepdims=True)
		overlap_count = tf.reduce_sum(tf.cast(overlap_map, tf.float32), [1, 2], keepdims=True)
	
	except:
		
		gt_count = tf.reduce_sum(tf.cast(y_true, tf.float32), [1, 2], keep_dims=True)
		pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32), [1, 2], keep_dims=True)
		overlap_count = tf.reduce_sum(tf.cast(overlap_map, tf.float32), [1, 2], keep_dims=True)

	iou = tf.div(overlap_count, gt_count + pred_count - overlap_count)

	return iou
	'''
	#计算heatmap与label相乘后非0点个数
	pred_0 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(y_true, 0), tf.int64), tf.cast(tf.equal(y_pred, 0), tf.int64)))

	pred_1 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(y_true, 1), tf.int64), tf.cast(tf.equal(y_pred, 1), tf.int64)))
	pred_2 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(y_true, 2), tf.int64), tf.cast(tf.equal(y_pred, 2), tf.int64)))
	pred_3 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(y_true, 3), tf.int64), tf.cast(tf.equal(y_pred, 3), tf.int64)))
	pred_4 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(y_true, 4), tf.int64), tf.cast(tf.equal(y_pred, 4), tf.int64)))
	#标签中大于0的点的个数
	gt_all = tf.count_nonzero(tf.cast(tf.greater(y_true, 0), tf.int64))

	# Compute mIoU of Lanes 预测与标签的iou
	overlap_1 = pred_1
	union_1 = tf.add(tf.count_nonzero(tf.cast(tf.equal(y_true, 1), 
												tf.int64)), 
						tf.count_nonzero(tf.cast(tf.equal(y_pred, 1), 
												tf.int64)))
	union_1 = tf.subtract(union_1, overlap_1)
	IoU_1 = tf.divide(overlap_1, union_1)

	overlap_2 = pred_2
	union_2 = tf.add(tf.count_nonzero(tf.cast(tf.equal(y_true, 2), 
												tf.int64)), 
						tf.count_nonzero(tf.cast(tf.equal(y_pred, 2), 
												tf.int64)))
	union_2 = tf.subtract(union_2, overlap_2)
	IoU_2 = tf.divide(overlap_2, union_2)

	overlap_3 = pred_3
	union_3 = tf.add(tf.count_nonzero(tf.cast(tf.equal(y_true, 3), 
												tf.int64)), 
						tf.count_nonzero(tf.cast(tf.equal(y_pred, 3), 
												tf.int64)))
	union_3 = tf.subtract(union_3, overlap_3)
	IoU_3 = tf.divide(overlap_3, union_3)

	overlap_4 = pred_4
	union_4 = tf.add(tf.count_nonzero(tf.cast(tf.equal(y_true, 4), 
												tf.int64)), 
						tf.count_nonzero(tf.cast(tf.equal(y_pred, 4), 
												tf.int64)))
	union_4 = tf.subtract(union_4, overlap_4)
	IoU_4 = tf.divide(overlap_4, union_4)

	IoU = tf.reduce_mean(tf.stack([IoU_1, IoU_2, IoU_3, IoU_4]))

	return IoU
	
class UNet(object):
	
	def __init__(self, max_iter, batch_size=32, init_lr=0.004, power=0.9, momentum=0.9, stddev=0.02, regularization_scale=0.0001, alpha=0.25, gamma=2.0, fl_weight=0.1):
		
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.init_lr = init_lr
		self.power = power
		self.momentum = momentum
		self.stddev = stddev
		self.regularization_scale = regularization_scale
		self.alpha = alpha
		self.gamma = gamma
		self.fl_weight = fl_weight
		self.graph = tf.Graph()
		
		with self.graph.as_default():
			
			self.X = tf.placeholder(tf.float32, shape=(self.batch_size, cfg.width, cfg.height, 3))
			self.Y = tf.placeholder(tf.float32, shape=(self.batch_size, cfg.width, cfg.height, cfg.num_class))

			self.build_arch()

			if cfg.is_quantize:
				g = tf.get_default_graph()
				if cfg.is_training:
					tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=4000) 
				else:
					tf.contrib.quantize.create_eval_graph(input_graph=g)

			self.evaluation()

			if cfg.is_training:
				self.loss()
				self._summary()
				self.global_iter = tf.Variable(0, name='global_iter', trainable=False)
				self.lr = tf.train.polynomial_decay(self.init_lr, self.global_iter, self.max_iter, end_learning_rate=1e-8, power=self.power)
				#self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
				self.optimizer = tf.train.AdamOptimizer(self.lr, cfg.beta1, cfg.beta2)

				self.train_op = slim.learning.create_train_op(self.total_loss,
                                                      self.optimizer,
                                                      global_step=self.global_iter)
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(update_ops):
					updates = tf.group(*update_ops)
					self.train_op = control_flow_ops.with_dependencies([updates], self.train_op)
				'''
				var_list = tf.trainable_variables()
				g_list = tf.global_variables()
				bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
				bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
				var_list += bn_moving_vars
				

				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				self.train_op = slim.learning.create_train_op(self.total_loss,
                                                          self.optimizer,
                                                          global_step=self.global_iter)
                                                          #variables_to_train=var_list)
				'''
				self.saver = tf.train.Saver()
			else:
				self.saver = tf.train.Saver()

			if cfg.is_training:
				model_save_dir = './models'
				self.write_graph = tf.train.write_graph(graph_or_graph_def=tf.get_default_graph(), logdir='',
                             name='{:s}/unet_struct.pb'.format(model_save_dir))

		tf.logging.info('Setting up the main structure')
	
	def build_arch(self):
		with tf.variable_scope('unet'):
			self.o = unet(self.X)

	def loss(self):
		
		######### -*- Softmax Loss -*- #########
		self.softmax_fuse, self.cefuse = pw_softmaxwithloss_2d(self.Y, self.o)
		self.total_ce = self.cefuse
		
		######### -*- Focal Loss -*- #########
		#self.fl = focal_loss(self.Y, self.o, alpha=self.alpha, gamma=self.gamma)
		
		######### -*- Total Loss -*- #########
		#self.total_loss = self.total_ce + self.fl_weight * self.fl
		self.total_loss = self.total_ce
	
	def evaluation(self):
		
		self.prediction = tf.argmax(self.o, axis = 3)
		self.ground_truth = tf.argmax(self.Y, axis = 3)
		self.iou = cal_iou(self.ground_truth, self.prediction)
		self.mean_iou = self.iou
	
	def _summary(self):
		
		trainval_summary = []
		trainval_summary.append(tf.summary.scalar('softmax_loss', self.total_ce))
		#trainval_summary.append(tf.summary.scalar('focal_loss', self.fl))
		trainval_summary.append(tf.summary.scalar('total_loss', self.total_loss))
		trainval_summary.append(tf.summary.scalar('mean_iou', self.mean_iou))
		self.trainval_summary = tf.summary.merge(trainval_summary)
