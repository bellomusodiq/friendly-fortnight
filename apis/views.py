from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from io import BytesIO
import base64
from models.mnist_cnn import predict

# Create your views here.


class MNISTAPIView(APIView):

    def post(self, request):
        im = Image.open(BytesIO(base64.b64decode(request.data['data']))).convert('1').resize((28, 28))
        prob, prediction = predict(im)
        distribution = {}
        for i in range(10):
            distribution[i] = '{:.2f}'.format(prob[i] * 100)
        
        return Response({'prob': distribution, 'prediction': prediction})
