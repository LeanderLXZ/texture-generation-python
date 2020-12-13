import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def generate_input_image(noise = True, content_img = None, imsize = imsize):
    """ Used to generate input image

    input:
        noise (bool): used noise as input or not
        content_img: the content image
        imsize (int): size of the output image

    output:
        input_img: the input image
    """
    if noise:
        return torch.randn((1, 3, imsize, imsize), device=device)
    else:
        if content_img == None:
            raise ('Please input content image')
        else:
            return content_img.clone()


if __name__ == '__main__':
    pass