from model.cnn_model import *
from model.cnn_image import *
from model.evaluation import *

import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import copy
from os.path import join
import glob

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)
    return path


def generate_compare(cnn,imgs_pair,saving_path):
    loss = 0
    for i in imgs_pair:
        
        l = evaluation(cnn,i[0],i[1])
        loss +=l
        temp = i[0].split("/",-1)
        name = temp[-1][:-6]
        print(name)
        with open(join(saving_path, 'npscores','scores'),'a+') as f:
            f.write(name + ' ' + str(loss) + '\n')
        loss = 0
        print('data written')


if __name__ == '__main__':
    # import pre-trained model 
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    #get result images:
    g='../data/results_np'
    imgs_original = []
    imgs_generated = []
    x = glob.glob(os.path.join(g,'*'))
    for fname in x:
        for iname in os.listdir(fname):
            if iname[-5] == '0':
                imgs_original.append(join(fname,iname))
            elif int(iname[-5]) > 0:
                imgs_generated.append(join(fname,iname))
    imgs_pair = list(zip(imgs_original,imgs_generated))


    mkdir('../results/npscores/')

    # clean scores file
    if os.path.isfile('../results/npscores/scores'):
        os.remove('../results/npscores/scores')

    # compare generated
    generate_compare(cnn,imgs_pair,saving_path = '../results/',)

  