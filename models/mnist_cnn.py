import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ml_backend.settings import BASE_DIR

mean_gray = 0.1307
stdev_gray = 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean_gray,), (stdev_gray,))])

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, stride=1, padding=1, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.cnn2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.fc1(x.view(-1, 1568))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN()
model.load_state_dict(torch.load(BASE_DIR / 'models' / 'weights' / 'mnist_cnn.tar'))

def predict(image):
    with torch.no_grad():
        model.eval()
        img = transform(image)
        img_np = np.array(img)
        img_np = img_np.reshape(28, 28, 1)
        img.unsqueeze_(0)
        output = model(img)
        _, prediction = torch.max(output, 1)
        return list(F.softmax(output, 1).numpy()[0]), prediction.item()
