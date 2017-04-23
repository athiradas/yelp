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
import urllib.request

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


class load_image:
	def from_url(self, url):
		if url == None:
			pass
		else:
			urllib.request.urlretrieve(url, os.path.join(settings.BASE_DIR+'/labels/static/photos/1234.jpg'))
			import csv

			f = open('new_id.csv', 'wt')
			try:
				writer = csv.writer(f)
				writer.writerow( ('business_id', 'photo_id') )
				writer.writerow( ('9999', '1234') )
			finally:
				f.close()

			

class feature_extraction:

	def PreprocessImage(img_path, show_img=False,invert_img=False):
				img = io.imread(str(img_path))
				if(invert_img):
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
				
	def get_imlist(self, path):
		#Returns a list of filenames for all jpg images in a directory. 

		return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
  
	def inception_7(self):
		start_time =  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		prefix = os.path.join(settings.BASE_DIR+"/labels/model/inception_7/Inception-7")
		num_round = 1
		network = model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=bs)
		inner = network.symbol.get_internals()
		inner_feature = inner['flatten_output']
		fea_ext = model.FeedForward(ctx=mx.cpu(),symbol=inner_feature,numpy_batch_size=bs,arg_params=network.arg_params,aux_params=network.aux_params,allow_extra_params=True)
		img_path = os.path.join(settings.BASE_DIR+'/labels/static/photos/')
		images = self.get_imlist(img_path)
		img_count = len(glob.glob1(img_path,"*.jpg"))		
		feat_holder = np.zeros([img_count,2048])
		for num_ph, image in enumerate(images):
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
		np.save(os.path.join(settings.BASE_DIR+'/labels/static/photos/new.npy'),feat_holder)
		
		
		
		
		for nb,lb in enumerate(labels):
			train_cl = pd.read_csv('train_labels_cl.csv')
		
			train_cl = dict(np.array(train_cl[['business_id',lb]]))
		
			biz_features['lb'] = biz_features['business_id'].apply(lambda x: train_cl[x])
				
			
			df_train_values = biz_features['lb']
			
			df_train_features = biz_features.drop(['business_id','lb'],axis=1)
			
			xg_train = xgb.DMatrix(df_train_features, label=df_train_values)

			bst = xgb.train(param, xg_train,iter_label[lb])

			model_dict[lb] = bst
			
			df_train_features = None
			
			df_test_features = None
			
			xg_train = None

		
		
		
		
		
		
		
		
		labels = ['label_'+str(i) for i in range(9)]
		
		test_to_biz = pd.read_csv('new_id.csv')

		test_image_features = np.load('new.npy',mmap_mode='r')
		 
		test_image_id = list(np.array(test_to_biz['photo_id'].unique()))
		 
		uni_bus = test_to_biz['business_id'].unique()
		 
		coll_arr = np.zeros([len(uni_bus),2048])
		 
		for nb,ub in enumerate(uni_bus):
			if(nb%1000==0):
				print(nb)
			image_ids = test_to_biz[test_to_biz['business_id']==ub]['photo_id'].tolist()  
			image_index = [test_image_id.index(x) for x in image_ids]
			features = test_image_features[image_index]
			x1 = np.mean(features,axis=0)
			x1 = x1.reshape([1,2048])
			coll_arr[nb,:] = x1

			
		biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
		
		coll_arr = pd.DataFrame(coll_arr)
		
		frames = [biz_features,coll_arr]
		
		biz_features = pd.concat(frames,axis=1)
				
		coll_arr = pd.DataFrame(coll_arr)
    
		frames = [biz_features,coll_arr]
		
		biz_features = pd.concat(frames,axis=1)
		
		del coll_arr,frames,test_to_biz,test_image_features,test_image_id,image_ids,image_index,features
		model_dict = {}
		
		result = np.zeros([biz_features.shape[0],9])
		
		for nb,lb in enumerate(labels):
			
			print('predicting',lb)
			df_test_features = biz_features.drop(['business_id'],axis=1)
			
			bst = model_dict[lb]
			
			yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]
			
			result[:,nb] = yprob
		
		end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		
		return (self.get_imlist(os.path.join(settings.BASE_DIR+'/labels/static/photos/')))

		return image
		



