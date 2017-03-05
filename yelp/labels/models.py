from __future__ import unicode_literals

from django.db import models

# Create your models here.


class Preprocess(models.Model):

	def PreprocessImage(path, show_img=False,invert_img=False):
		img = io.imread(path)
		if(invert_img):
			img = np.fliplr(img)
		short_egde = min(img.shape[:2])
		yy = int((img.shape[0] - short_egde) / 2)
		xx = int((img.shape[1] - short_egde) / 2)
		crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
		# resize to 224, 224
		resized_img = transform.resize(crop_img, (299, 299))
		if show_img:
			io.imshow(resized_img)
		# convert to numpy.ndarray
		sample = np.asarray(resized_img) * 256
		# swap axes to make image from (299, 299, 3) to (3, 299, 299)
		sample = np.swapaxes(sample, 0, 2)
		sample = np.swapaxes(sample, 1, 2)
		# sub mean 
		normed_img = sample - 128
		normed_img /= 128.
		return np.reshape(normed_img,(1,3,299,299))
	
	def inception_7(self):
		start_time =  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		prefix = "static/model/inception_7/"
		num_round = 1
		network = model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=bs)
		inner = network.symbol.get_internals()
		inner_feature = inner['flatten_output']
		fea_ext = model.FeedForward(ctx=mx.cpu(),symbol=inner_feature,numpy_batch_size=bs,arg_params=network.arg_params,aux_params=network.aux_params,allow_extra_params=True)
		biz_ph = pd.read_csv('sample.csv')
		ph = biz_ph['photo_id'].unique().tolist()		 
		feat_holder = np.zeros([len(ph),2048])
		for num_ph,photo in enumerate(ph):
			fp = 'static/photos/'+str(photo)+'.jpg'
			try:
				feat_holder[num_ph,:]=fea_ext.predict(PreprocessImage(fp,show_img=False,invert_img=False))
			except FileNotFoundError:
				pass
		np.save('sample.npy',feat_holder)
		end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
		return 'inception_7'

