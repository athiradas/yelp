from __future__ import unicode_literals

from django.db import models
from django.conf import settings

# Create your model models here.

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

import random



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
import xgboost as xgb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adadelta, Adagrad, Adam,RMSprop
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
import sklearn

from sklearn.cluster import MiniBatchKMeans

import tensorflow as tf

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

			f = open(os.path.join(settings.BASE_DIR+'/labels/static/photos/new_id.csv'), 'wt')
			try:
				writer = csv.writer(f)
				writer.writerow( ('business_id', 'photo_id') )
				writer.writerow( ('0', '1234') )
			finally:
				f.close()

			

class feature_extraction:

	def get_net(self, num_feat,num_out):
		clf = Sequential()
		clf.add(Dense(input_dim=num_feat, output_dim=65))
		clf.add(PReLU())
		clf.add(Dense(input_dim = 65, output_dim = 35))
		clf.add(PReLU())
		clf.add(Dense(input_dim = 35, output_dim=num_out, activation='sigmoid'))
		clf.compile(optimizer=Adam(), loss='categorical_crossentropy')
		return clf
	
	
	def np_thresh(self, x1,thresh):
    
		x1 = x1-thresh
		
		x1 = x1>0
		
		x1 = np.array(x1,dtype=int)
		
		return x1
	
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
		random.seed(42)
		np.random.seed(100)
		tf.set_random_seed(1)
		


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
		
		
		
		labels = ['label_'+str(i) for i in range(9)]
		
		param = {}
		param['objective'] = 'multi:softprob'
		param['eta'] = 0.1
		param['max_depth'] = 3
		param['subsample'] = 0.6
		param['silent'] = 1
		param['nthread'] = 4
		param['eval_metric'] = "mlogloss"
		num_round = 100
		param['num_class'] = 2
		param['seed']=0

		iter_label = {'label_1': 6, 'label_2': 1, 'label_6': 11, 'label_5': 12, 'label_3': 3, 'label_7': 12, 'label_0': 14, 'label_4': 19, 'label_8': 10}
		
		param['subsample'] = param['subsample']*0.8
		train_to_biz = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_id.csv'))

		train_image_features = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/train_inception_7.npy'),mmap_mode='r')
		
		uni_bus = train_to_biz['business_id'].unique()
		
		coll_arr = np.zeros([len(uni_bus),2048])
		
		for nb,ub in enumerate(uni_bus):
			if(nb%1000==0):
				print(nb)
			tbz = np.array(train_to_biz['business_id']==ub,dtype=bool)
			x1 = np.array(train_image_features[tbz,:])
			x1 = np.mean(x1,axis=0)
			x1 = x1.reshape([1,2048])
			coll_arr[nb,:] = x1
			
		biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
		
		coll_arr = pd.DataFrame(coll_arr)
		
		frames = [biz_features,coll_arr]
		
		biz_features = pd.concat(frames,axis=1)
		
		#del train_to_biz,train_image_features,coll_arr,frames
		
		
		model_dict = {}
		
		for nb,lb in enumerate(labels):
			train_cl = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_labels_cl.csv'))
		
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
			
			
		test_to_biz = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/photos/new_id.csv'))

		test_image_features = np.load( os.path.join(settings.BASE_DIR+'/labels/static/photos/new.npy'),mmap_mode='r')
		 
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
		
		#del coll_arr,frames,test_image_features,test_image_id,image_ids,image_index,features
		
		result = np.zeros([biz_features.shape[0],9])
		
		for nb,lb in enumerate(labels):
			
			print('predicting',lb)
			df_test_features = biz_features.drop(['business_id'],axis=1)
			
			bst = model_dict[lb]
			
			yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]
			
			result[:,nb] = yprob
		
		np.save(os.path.join(settings.BASE_DIR+'/labels/static/data/Model1_Full_result.npy'),result)



		
###############MODEL 2 ############################

#train_image_features = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/train_inception_7.npy'),mmap_mode = 'r')

		test_image_features = np.load(os.path.join(settings.BASE_DIR+'/labels/static/photos/new.npy'),mmap_mode = 'r')

		tr_ts = np.vstack((train_image_features,test_image_features))

		np.save( os.path.join(settings.BASE_DIR+'/labels/static/photos/train_test_kmn.npy'),tr_ts)
		
		train_image_features = np.load(os.path.join(settings.BASE_DIR+'/labels/static/photos/train_test_kmn.npy'),mmap_mode = 'r')
		
		
		

		for num_cluster in ([2,3,4,5]):
		
			kmn_holder = MiniBatchKMeans(n_clusters=num_cluster)

			kmn = kmn_holder.fit_predict(train_image_features[:,:])

			kmn_train = kmn[:57870]

			kmn_test = kmn[57870:]
			
			coll_arr = np.zeros([len(uni_bus),(2048*num_cluster)])
			for nb,lb in enumerate(labels):
				
				print('predicting',lb)
				df_test_features = biz_features.drop(['business_id'],axis=1)
				
				bst = model_dict[lb]								
				bst.save_model('model1')
				bst = xgb.Booster(param)
				bst.load_model('model1')

				yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]
				
				result[:,nb] = yprob	
				np.save(os.path.join(settings.BASE_DIR+'/labels/static/data/Model1_Full_result.npy'),result)
				
			for nb,ub in enumerate(uni_bus):
				image_ids = test_to_biz[test_to_biz['business_id']==ub]['photo_id'].tolist()  
				image_index = [test_image_id.index(x) for x in image_ids]
				features = test_image_features[image_index]
				l1 = kmn_test[image_index]
				for kn in range(num_cluster):
					x2 = features[l1==kn]
					x2 = np.mean(x2,axis=0)
					x2= x2.reshape([1,2048])
					if(np.isnan(np.sum(x2))):
						coll_arr[nb,(2048*(kn)):(2048*(kn+1))] = np.zeros([1,2048])
					else:
						coll_arr[nb,(2048*(kn)):(2048*(kn+1))] = x2
				
			biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
			
			coll_arr = pd.DataFrame(coll_arr)
			
			frames = [biz_features,coll_arr]
			
			biz_features = pd.concat(frames,axis=1)
			
			result = np.zeros([biz_features.shape[0],9])
			
			for nb,lb in enumerate(labels):
				
				print('predicting',lb, num_cluster)
				df_test_features = biz_features.drop(['business_id'],axis=1)
				
				bst = model_dict[lb]
								
				bst.save_model('model2')
				bst = xgb.Booster(param)
				bst.load_model('model2')

				yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]

								
				#preds = bst.predict(dtest)
				
				result[:,nb] = yprob
			
			np.save(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_'+str(num_cluster)+'_result.npy') ,result)
			
#################FINAL MODEL ##################

		train_to_biz = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_labels.csv'))
    
		uni_bus = train_to_biz['business_id'].unique()

		blend1 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model1_Full.npy'))
    
		blend2 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_2.npy'))
		
		blend3 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_3.npy'))
		
		blend4 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_4.npy'))
		
		blend5 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_5.npy'))
		
		x1 = np.hstack((blend1,blend4,blend2,blend3,blend5))

		x1 = pd.DataFrame(x1)
		
		x1['business_id'] = uni_bus
		
		other_cols = ['business_id','label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8']

		
		for nb,lb in enumerate(labels):
			train_cl = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_labels_cl.csv'))
		
			train_cl = dict(np.array(train_cl[['business_id',lb]]))
		
			x1[lb] = x1['business_id'].apply(lambda x: train_cl[x])
			
		df_train_values = np.array(x1[labels])
    
		df_train_features = np.array(x1.drop(other_cols,axis=1))
		
		with tf.Session() as sess:
			bst = self.get_net(df_train_features.shape[1],df_train_values.shape[1])
					
				
			bst.fit(df_train_features,df_train_values,batch_size=30,nb_epoch=26)
			
			df_train_features = None
			
			df_test_features = None
			
			xg_train = None
			
			
			# Predict on the test set
		
			test_to_biz = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/photos/new_id.csv'))
			
			uni_bus = test_to_biz['business_id'].unique()
				
			blend1 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model1_Full_result.npy'))
			
			blend2 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_2_result.npy'))
			
			blend3 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_3_result.npy'))
			
			blend4 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_4_result.npy'))
			
			blend5 = np.load(os.path.join(settings.BASE_DIR+'/labels/static/data/Model2_Full_5_result.npy'))
			
			x1 = np.hstack((blend1,blend4,blend2,blend3,blend5))  
			
			x1 = pd.DataFrame(x1)
			
			x1['business_id'] = uni_bus
			
			result = np.zeros([x1.shape[0],9])

			df_test_features = np.array(x1.drop(['business_id'],axis=1))

			result = bst.predict((df_test_features))

			result2 = np.zeros(result.shape)
			f1_thresh = [0.43]*9
			
			for f1 in range(9):
			
				result2[:,f1] = self.np_thresh(result[:,f1],f1_thresh[f1])

			
			bid = np.array(x1['business_id'])
			
			fin = {}
			
			for i in range(result2.shape[0]):
				x = result2[i,:]
				li = [((q)) for q in range(9) if x[q]==1]
				fin[bid[i]] = li
				
			for j in fin.keys():
				fin[j] = ' '.join(str(e) for e in fin[j])
			
			x1 = pd.DataFrame(x1['business_id'])
			
			x1['labels'] = x1['business_id'].apply(lambda x: fin[x] if x in fin.keys() else '0')
			

			train_label = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_labels.csv'))

			train_to_biz = pd.read_csv(os.path.join(settings.BASE_DIR+'/labels/static/data/train_id.csv'))


			train_label['desired'] = train_label['labels'].str.contains('4' and '1' and '2')
			desired_businesses = train_label[train_label.desired==True].business_id.tolist()
			desired_photos = train_to_biz[train_to_biz.business_id.isin(desired_businesses)].photo_id.tolist()

			num_images_for_show = 5
			photos_to_show = np.random.choice(desired_photos,num_images_for_show**2)
			photo_reco = list()
			for x in range(num_images_for_show**2):
				print (photos_to_show[x])
				photo_reco.append(photos_to_show[x])
		return (result, photo_reco)
		
