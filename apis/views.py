from models.transformers import transform_image
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from io import BytesIO
import base64
from models.mnist_cnn import predict
import json
from ml_backend.settings import BASE_DIR
import random
import string

# Create your views here.


def generate_string(length):

    letters = string.ascii_lowercase  # define the lower case string
    # define the condition for random.choice() method
    result = ''.join((random.choice(letters)) for x in range(length))
    return result


class MNISTAPIView(APIView):

    def post(self, request):
        im = Image.open(BytesIO(base64.b64decode(request.data['data']))).convert(
            '1').resize((28, 28))
        prob, prediction = predict(im)
        distribution = {}
        for i in range(10):
            distribution[i] = '{:.2f}'.format(prob[i] * 100)

        return Response({'prob': distribution, 'prediction': prediction})


class ImageAgumentation(APIView):

    def post(self, request):
        transforms = request.data.get('transforms')
        transforms = json.loads(transforms)
        img = request.FILES.get('file')
        protocol = 'http://'
        if request.is_secure():
            protocol = 'https://'
        host = request.META['HTTP_HOST']
        response = []
        with Image.open(img) as im:
            for i, result in enumerate(transform_image(im, transforms)):
                slug = generate_string(10)
                result['image'].save(
                    BASE_DIR / 'media' / 'transforms' / '{}-{}.jpg'
                    .format(result['transform'], slug
                            ), 'JPEG'
                )
                image_url = '{}{}/media/transforms/{}-{}.jpg'.format(
                    protocol, host, result['transform'], slug)
                response.append(
                    {'id': i, 'transform': result['transform'], 'image': image_url})
        return Response(response)
