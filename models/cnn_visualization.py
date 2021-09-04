from ml_backend.utils import generate_string
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
from ml_backend.settings import BASE_DIR

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                      std=[0.229, 0.224, 0.225])
])

vgg = models.vgg16(pretrained=True)

CUDA = torch.cuda.is_available()
if CUDA:
    vgg = vgg.cuda()

def visualize(image):
    with torch.no_grad():
        image = transform(image)
        image = image.unsqueeze(0)
        output = vgg(image)
        output = output.squeeze(0)

        module_list = list(vgg.features.modules())
        outputs = []
        names = []
        for layer in module_list[1:11]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            # Convert the 3D tensor to 2D and sum the same element of every channel
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale/feature_map.size(0)
            processed.append(gray_scale.data.cpu().numpy())

        fig = plt.figure(figsize=(40,40))

        for i in range(len(processed)):
            a = fig.add_subplot(10, 1, i+1)
            imgplot = plt.imshow(processed[i])
            plt.axis('off')
            a.set_title(names[i].split('(')[0], fontsize=30)

            file_name = 'f-map{}.jpg'.format(generate_string(10))
                
        plt.savefig(BASE_DIR / 'media' / 'cnn-visualization' / file_name, bbox_inches='tight')

        return file_name