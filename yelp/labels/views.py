from django.http import HttpResponse
from .models import Preprocess
from django.template import loader, RequestContext

def index(request):
        preprocess = Preprocess()
        preprocess_iv3 = preprocess.inception_v3()
        template = loader.get_template('labels/index.html')
        context = {'preprocess_iv3':preprocess_iv3}
        return HttpResponse(template.render(context, request))

