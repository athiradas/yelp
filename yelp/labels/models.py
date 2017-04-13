from __future__ import unicode_literals

from django.db import models
from django.conf import settings

# Create your models here.

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:23:46 2017

@author: athir
"""

'''
Using a pretrained net to generate features for images
Assuming the script is run on a GPU, it takes close to 6 hours to complete (This can be optimized by changing batch size)
The output files are close to 4GB each for train and test
The inception net weights and architecture need to be downloaded from here: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md
'''


import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
from mxnet import model
import pandas as pd
import time
import datetime
import os 
import glob

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

bs = 1

class Preprocess(models.Model):	
	def PreprocessImage(img_path, show_img=False,invert_img=False):
				#img = io.imread(str(img_path))
				'''if(invert_img):
					img = np.fliplr(img)
				short_egde = min(img.shape[:2])
				yy = int((img.shape[0] - short_egde) / 2)
				xx = int((img.shape[1] - short_egde) / 2)
				crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
				# resize to 224, 224
				resized_img = transform.resize(crop_img, (299, 299))
				# convert to numpy.ndarray
				sample = np.asarray(resized_img) * 256
				# swap axes to make image from (299, 299, 3) to (3, 299, 299)
				sample = np.swapaxes(sample, 0, 2)
				sample = np.swapaxes(sample, 1, 2)
				# sub mean 
				normed_img = sample - 128
				normed_img /= 128.
				return np.reshape(resized_img,(1,3,299,299)) #np.reshape(normed_img,(1,3,299,299))
				'''
				return (img_path)
			

class feature_extraction:
	def get_imlist(self, path):
		#Returns a list of filenames for all jpg images in a directory. 

		return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
  
	def inception_7(self):
		start_time =  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		preprocess = Preprocess()
		prefix = os.path.join(settings.BASE_DIR+"/labels/model/inception_7/Inception-7")
		num_round = 1
		network = model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=bs)
		inner = network.symbol.get_internals()
		inner_feature = inner['flatten_output']
		fea_ext = model.FeedForward(ctx=mx.cpu(),symbol=inner_feature,numpy_batch_size=bs,arg_params=network.arg_params,aux_params=network.aux_params,allow_extra_params=True)
		#biz_ph = pd.read_csv('sample.csv')
		#ph = biz_ph['photo_id'].unique().tolist()	
		img_path = os.path.join(settings.BASE_DIR+'/labels/static/photos/')
		images = self.get_imlist(img_path)
		img_count = len(glob.glob1(img_path,"*.jpg"))		
		feat_holder = np.zeros([img_count,2048])
		for num_ph, image in enumerate(images):
					'''try:
						feat_holder[img_count,:]=fea_ext.predict(preprocess.PreprocessImage(fp))
					except FileNotFoundError:
						pass '''
					img = io.imread(image)
					short_egde = min(img.shape[:2])
					yy = int((img.shape[0] - short_egde) / 2)
					xx = int((img.shape[1] - short_egde) / 2)
					crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
					# resize to 224, 224
					resized_img = transform.resize(crop_img, (299, 299))
					# convert to numpy.ndarray
					sample = np.asarray(resized_img) * 256
					# swap axes to make image from (299, 299, 3) to (3, 299, 299)
					sample = np.swapaxes(sample, 0, 2)
					sample = np.swapaxes(sample, 1, 2)
					# sub mean 
					normed_img = sample - 128
					normed_img /= 128.
					img_process = np.reshape(resized_img,(1,3,299,299))
					feat_holder[num_ph,:]=fea_ext.predict(img_process)
					#img_process = preprocess.PreprocessImage(str(image))
					
		np.save('new.npy',feat_holder)
		end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		return (start_time, end_time)
		

