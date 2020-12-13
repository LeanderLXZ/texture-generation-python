from model.cnn_image import *
from model.cnn_model import *

import time

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired depth layers to compute style/content losses:
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def gram_matrix(x):
    """ Calculate the gram martix based on the input tensor

    input:
        x (tensor): size: b, c, h, w

    output:
        gram (tensor): size: b*c, b*c
    """
    # get input's dimension
    b, c, h, w = x.size()

    # calculate the gram martix
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())

    return G.div(b * c * h * w)


def evaluation(pre_model, img_1, img_2,
                   default_mean_std = True,
                   style_layers=default_style_layers,
                   weight = 1000000):
    """ evaluate the style loss between two input images

    input:
        pre_model: used to evaluate the style 

    output:
        style_losses (int): the style difference between the two input images
    """
    # load the image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    img_1 = image_loader(img_1)
    img_2 = image_loader(img_2)

    cnn = copy.deepcopy(pre_model)

    # normalization module
    normalization = Normalization(default_mean_std = default_mean_std)

    style_losses = 0

    # create our model
    model = nn.Sequential(normalization)

    # increment every time we see a conv
    i = 0  
    # go through all the layers
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # According to Alexis Jacq, the in-place version doesn't play 
            # very nicely with the ContentLoss with the ContentLoss and StyleLoss 
            # we insert below. So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'maxpool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            # calculate target style
            style_1 = model(img_1).detach()
            style_1 = gram_matrix(style_1)
            style_2 =  model(img_2).detach()
            style_2 = gram_matrix(style_2)
            # save the loss
            style_losses += F.mse_loss(style_1, style_2) / len(style_layers)
    
    style_losses *= weight
    return float(style_losses)


if __name__ == '__main__':
    pass