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


from django.http import HttpResponse
from .models import feature_extraction, load_image
from django.template import loader, RequestContext


def index(request):
	template = loader.get_template('labels/index.html')
	if request.method == 'GET':
			url = request.GET.get('url', True)
			if url == True:
					url = None #"Enter URL of the image"
					return HttpResponse(template.render(request))
			else:
				li = load_image()
				li.from_url(url)
				fe = feature_extraction()
				label_prob = fe.inception_7()
				label = {}
				print (label_prob)
				label["Good for Lunch"] = label_prob[0][0] * 100
				label["Good for Dinner"] = label_prob[0][1] * 100
				label["Takes Reservations"] = label_prob[0][2] * 100
				label["Outdoor Seating"] = label_prob[0][3] * 100
				label["Restaurant is Expensive"] = label_prob[0][4] * 100
				label["Has Alcohol"] = label_prob[0][5] * 100
				label["Has Table Service"] = label_prob[0][6] * 100
				label["Ambience is Classy"] = label_prob[0][7] * 100
				label["Good for Kids"] = label_prob[0][8] * 100
				
				true_label = {k: v for k, v in label.items() if v > 45}
				context = {'true_label':true_label, 'url': url}
				print (true_label)
				return HttpResponse(template.render(context, request))




