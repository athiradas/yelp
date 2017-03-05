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

from django.http import HttpResponse
from .models import Preprocess
from django.template import loader, RequestContext

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

bs = 1

def index(request):
        preprocess = Preprocess()
        preprocess_iv7 = preprocess.inception_7()
        template = loader.get_template('labels/index.html')
        context = {'preprocess_i7':preprocess_i7}
        return HttpResponse(template.render(context, request))

