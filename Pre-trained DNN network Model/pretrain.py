import argparse

import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

import cv2 as cv

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
opt = parser.parse_args()

classes = 10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

context = [mx.cpu()]

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
kwargs = {'classes': classes, 'pretrained': pretrained}
net = get_model(model_name, **kwargs)

if not pretrained:
    net.load_parameters(opt.saved_params, ctx = context)

# Load Images
img = image.imread(opt.input_pic)

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cvImg = cv.imread(opt.input_pic)

print("Kernel Convolution\n=====================")
# blur
for ind in range(4):
    kernel_size = 3 + 2 * ind
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= (kernel_size * kernel_size)

    dst = cv.filter2D(cvImg, -1, kernel)
    img = transform_fn(mx.nd.array(dst))
    pred = net(img.expand_dims(0))

    ind = nd.argmax(pred, axis=1).astype('int')
    print('The input picture is classified to be [%s].'%
          (class_names[ind.asscalar()]))

print("Kernel Convolution\n=====================")

print("Brightness and Contrast\n=====================")

# brightness and contrast
for alpha in range(1, 2):
    for beta in range(0, 30, 5):
        dst = cv.addWeighted(cvImg, alpha, np.zeros(cvImg.shape, cvImg.dtype), 0, beta)
        img = transform_fn(mx.nd.array(dst))
        pred = net(img.expand_dims(0))

        ind = nd.argmax(pred, axis=1).astype('int')
        print('The input picture is classified to be [%s].'%
              (class_names[ind.asscalar()]))

print("Brightness and Contrast\n=====================")
    
# img = transform_fn(img)
# pred = net(img.expand_dims(0))

# ind = nd.argmax(pred, axis=1).astype('int')
# print('The input picture is classified to be [%s], with probability %.3f.'%
#       (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))
