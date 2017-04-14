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
                        fe = feature_extraction()
                        li = load_image()
                        li.from_url(url)
                        preprocess_i7 = 'Not a new url'#fe.inception_7()
                        context = {'preprocess_i7':preprocess_i7, 'url': url}
                        return HttpResponse(template.render(context, request))


