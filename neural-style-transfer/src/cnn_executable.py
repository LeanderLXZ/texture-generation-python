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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output im
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)
    return path

def get_imgs_name(file):
    """ Get images name form a txt file

    input:
        file (str): name of the txt file
        path: prefix of the system path

    output:
        names (list): the name of this type of img
        imgs (list): a list store the path of the image
    """
    styles = []
    names = []
    imgs = []

    with open(file, 'r') as f:
        for line in f.readlines():
            s = line.strip('\n').split('/')
            styles.append(s[0])
            s = ''.join(s[1:])
            imgs.append(s)
            names.append(s[:-4])

    return styles, names, imgs

def tansfer(cnn, content_img, style_name, style_img, file,
            num_output = 5,
            default_content_layers = ['conv_4'],
            default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    """
    
    input:
        cnn: the pre train model.
        content_img (tensor)
        style_name (str): like 'blotchy_0003'
        style_img (tensor)
    output:
        out_imgs: the group of new generation pictures
    """
    out_imgs = []
    scores = 0

    # store the style image
    org_image = copy.deepcopy(style_img)
    torchvision.utils.save_image(org_image, file + '/' + style_name + '_0.jpg')

    plt.figure()
    times = []
    losses = []
    for i in range(num_output):
        input_img = generate_input_image(noise = True, content_img=content_img)

        output, g, time, s_losses, c_losses, t_losses = \
            style_transfer(cnn, content_img, style_img, input_img,
                            default_mean_std = False, 
                            num_steps=300, 
                            style_weight=100000, 
                            content_weight=0
                            )
        times.append(time)
        losses.append([s_losses, c_losses, t_losses])

        # save fig
        out_imgs.append(output)
        save_name = '{}_{}'.format(style_name, i + 1)
        torchvision.utils.save_image(output, file + '/{}.jpg'.format(save_name))
        
    losses = np.mean(np.array(losses), axis=0)
    
    # plt.figure()
    # plt.plot(losses[0])
    # plt.plot(losses[1])
    # plt.plot(losses[2])
    # plt.title('Model Loss - {}'.format(style_name))
    # plt.ylabel('Loss')
    # plt.xlabel('Step')
    # plt.legend(['Style Loss', 'Content Loss', 'Total Loss'], loc='upper right')
    # plt.savefig(file + '/{} losses'.format(style_name))

    plt.figure()
    plt.plot(losses[0])
    plt.title('Model Loss - {}'.format(style_name))
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.savefig(file + '/{} losses'.format(style_name))

    #times = np.array(times)
    times = np.mean(times, 0)
    # plot the time
    plt.figure()
    plt.plot(range(len(times))[::10], times[::10])
    plt.xlabel('Iteration Times')
    plt.ylabel('Running Time')
    plt.title(style_name)
    plt.legend()
    plt.savefig(file + '/{} time'.format(style_name))

    return out_imgs

def worker(params):
    style = params['style']
    name = params['name']
    img = params['img'] 
    cnn = params['cnn']
    img_path = params['img_path']
    content_img = params['content_img']
    num_output = params['num_output']
    saving_path = params['saving_path']

    #this_saving_path = saving_path + '/' + name
    style_name = style
    img_name = name

    style_img = image_loader(img_path + style + '/'+ img)

    save_folder = mkdir(saving_path + img_name)

    out_images = tansfer(cnn, 
                            content_img, 
                            img_name, 
                            style_img,
                            save_folder,num_output)

    loss = 0

    g = str('../results/'+img_name)
    
    img_original = None
    imgs_generated = []
    for fname in os.listdir(g):
        if fname[-5] == '0':
            img_original = join(g, fname)
        elif fname[-5] not in ['e', 'c', 's'] and int(fname[-5]) > 0:
            imgs_generated.append(join(g, fname))

    for out in imgs_generated:
        l = evaluation(cnn, img_original, out)
        loss += l
        
    loss /= len(out_images)
    loss = round(loss, 4)

    with open(join(saving_path, 'scores','scores'),'a+') as f:
        f.write(img_name + ' ' + str(loss) + '\n')

if __name__ == '__main__':
    # import pre-trained model 
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # desired depth layers to compute style/content losses:
    default_content_layers = ['conv_4']
    default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    # set test datasets
    txt_file = '../data/test/test.txt'
    # read content image
    content_img = image_loader('../data/img/panda.jpg')

    mkdir('../results/scores/')

    # clean scores file
    if os.path.isfile('../results/scores/scores'):
        os.remove('../results/scores/scores')
    # generate and compare images:
    styles, names, imgs = get_imgs_name(txt_file)

    for style, name, img in zip(styles, names, imgs):
        params = {
            'style': style,
            'name': name,
            'img': img,
            'cnn': cnn,
            'content_img': content_img,
            'img_path': '../data/Textures/',
            'saving_path': '../results/',
            'num_output': 3
        }
        worker(params)
            