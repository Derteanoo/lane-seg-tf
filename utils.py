# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from PIL import Image
from config import cfg

# Take in image directory and return a directory containing image directory
# and images split into train, val, test
def create_image_lists(image_dir):
	
	result = {}
	
	training_images = []
	validation_images = []
	testing_images = []
	
	for category in ["train", "val", "test"]:
		
		category_path = os.path.join(image_dir, category)
		main_path = os.path.join(category_path, "main")
		segmentation_path = os.path.join(category_path, "segmentation")
		main_filenames = os.listdir(main_path)
		segmentation_filenames = os.listdir(segmentation_path)
		
		assert len(main_filenames) == len(segmentation_filenames), "The number of images in the " + main_path + " is not equal to that in the " + segmentation_path
		
		for main_filename in main_filenames:
			
			if category == "train":
				
				training_images.append(main_filename)
			
			if category == "val":
				
				validation_images.append(main_filename)
			
			if category == "test":
				
				testing_images.append(main_filename)
			
			else:
				
				pass
	
	result = {
		"root": image_dir,
		"train": training_images,
		"val": validation_images,
		"test": testing_images
	}
	
	return result

def get_batch_of_trainval(result, category="train", batch_size=32):
	
	assert category != "test", "category is not allowed to be 'test' here"
	
	image_dir = result["root"]
	filenames = result[category]
	batch_list = random.sample(filenames, batch_size)
	
	main_list = []
	segmentation_list = []
	
	for filename in batch_list:
		
		category_path = os.path.join(image_dir, category)
		main_path = os.path.join(category_path, "main/" + filename)
		segmentation_path = os.path.join(category_path, "segmentation/" + filename[:-4] + '_1.png')

		img = Image.open(main_path)
		w, h = img.size
		left, top, right, bottom = 0, cfg.horizon, w, h
		region = (left,top,right,bottom)
		img = img.crop(region).resize((cfg.width, cfg.height), Image.NEAREST)
		img = np.array(img, np.float32)
		img /= 127.5
		img -= 1.
		
		assert img.ndim == 3 and img.shape[2] == 3
		
		img = np.expand_dims(img, axis=0)

		label = Image.open(segmentation_path).convert("L").crop(region).resize((cfg.width, cfg.height), Image.NEAREST)
		label = np.array(label, np.int64)
		label = np.eye(cfg.num_class)[label]
		label = np.expand_dims(label, axis=0)
		
		main_list.append(img)
		segmentation_list.append(label)
	
	X = np.concatenate(main_list, axis=0)
	Y = np.concatenate(segmentation_list, axis=0)
	
	return X, Y

def get_batch_of_test(result, start_id, batch_size=32):
	
	image_dir = result["root"]
	filenames = result["test"]
	next_start_id = start_id + batch_size
	
	if next_start_id > len(filenames):
		
		next_start_id = len(filenames)
	
	paddings = start_id + batch_size - next_start_id
	
	main_list = []
	
	for idx in range(start_id, next_start_id):
		
		category_path = os.path.join(image_dir, "test")
		main_path = os.path.join(category_path, "main/" + filenames[idx])

		img = Image.open(main_path)
		w, h = img.size
		left, top, right, bottom = 0, cfg.horizon, w, h
		region = (left,top,right,bottom)
		img = img.crop(region).resize((cfg.width, cfg.height), Image.NEAREST)
		img = np.array(img, np.float32)
		img /= 127.5
		img -= 1.
		
		assert img.ndim == 3 and img.shape[2] == 3
		
		img = np.expand_dims(img, axis=0)
		
		main_list.append(img)
	
	for i in range(paddings):
		
		main_list.append(main_list[-1])
	
	X = np.concatenate(main_list, axis=0)
	
	return X, next_start_id, filenames[start_id:next_start_id]
