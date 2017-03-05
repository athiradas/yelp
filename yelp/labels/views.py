from django.http import HttpResponse
from .models import Preprocess
from django.template import loader, RequestContext

def index(request):
        preprocess = Preprocess()
        preprocess_iv7 = preprocess.inception_7()
        template = loader.get_template('labels/index.html')
        context = {'preprocess_i7':preprocess_i7}
        return HttpResponse(template.render(context, request))

