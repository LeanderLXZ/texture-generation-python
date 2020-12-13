import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

class ImageHelper(object):
    def save_image(self, plot_images, epoch):
        os.makedirs('cycle_gan_images', exist_ok=True)
        titles = ['Original', 'Transformed']
        fig, axs = plt.subplots(2, 2)
        cnt = 0
        for i in range(2):
            for j in range(2):
                axs[i,j].imshow(plot_images[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("cycle_gan_images/{}".format(epoch))
        plt.close()
        
    def plot20(self, images_paths_array):
        plt.figure(figsize=(10, 8))
        for i in range(20):
            img = plt.imread(images_paths_array[i])
            plt.subplot(4, 5, i+1)
            plt.imshow(img)
            plt.title(img.shape)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    def load_image(self, path):
        return np.asarray(Image.open(path))

    def imresize(self, image, size):
        im = Image.fromarray(image)
        return np.array(im.resize(size, Image.BICUBIC))
    
    def load_testing_image(self, path):
        self.img_res=(128, 128)
        path_X = glob(path + "/testA/*.jpg")
        path_Y = glob(path + "/testB/*.jpg")

        image_X = np.random.choice(path_X, 1)
        image_Y = np.random.choice(path_Y, 1)
        
        img_X = self.load_image(image_X[0])
        img_X = self.imresize(img_X, self.img_res)
        if np.random.random() > 0.5:
            img_X = np.fliplr(img_X)
        img_X = np.array(img_X)/127.5 - 1.
        img_X = np.expand_dims(img_X, axis=0)
        
        img_Y = self.load_image(image_Y[0])
        img_Y = self.imresize(img_Y, self.img_res)
        if np.random.random() > 0.5:
            img_X = np.fliplr(img_X)
        img_Y = np.array(img_Y)/127.5 - 1.
        img_Y = np.expand_dims(img_Y, axis=0)
        
        return (img_X, img_Y)
    
    def load_batch_of_train_images(self, path, batch_size=1):
        self.img_res=(128, 128)
        path_X = glob(path + "/trainA/*.jpg")
        path_Y = glob(path + "/trainB/*.jpg")
        
        self.n_batches = int(min(len(path_X), len(path_Y)) / batch_size)
        total_samples = self.n_batches * batch_size

        path_X = np.random.choice(path_X, total_samples, replace=False)
        path_Y = np.random.choice(path_Y, total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            batch_A = path_X[i*batch_size:(i+1)*batch_size]
            batch_B = path_Y[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.load_image(img_A)
                img_B = self.load_image(img_B)

                img_A = self.imresize(img_A, self.img_res)
                img_B = self.imresize(img_B, self.img_res)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B