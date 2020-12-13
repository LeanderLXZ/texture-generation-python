import numpy as np
from glob import glob
from PIL import Image
from cycle_gan import StyleStransfer

class ImageLoader(object):

    def __init__(self, image_size):
        self.image_size = image_size

    def load_image(self, path):
        return np.asarray(Image.open(path))

    def imresize(self, image, size):
        im = Image.fromarray(image)
        return np.array(im.resize(size, Image.BICUBIC))

    def get_test_image(self, path):

        path_X = glob(path + "/testA/*.jpg")
        path_Y = glob(path + "/testB/*.jpg")

        image_X = np.random.choice(path_X, 1)
        image_Y = np.random.choice(path_Y, 1)

        img_X = self.load_image(image_X[0])
        img_X = self.imresize(img_X, self.image_size)
        if np.random.random() > 0.5:
            img_X = np.fliplr(img_X)
        img_X = np.array(img_X)/127.5 - 1.
        img_X = np.expand_dims(img_X, axis=0)

        img_Y = self.load_image(image_Y[0])
        img_Y = self.imresize(img_Y, self.image_size)
        if np.random.random() > 0.5:
            img_X = np.fliplr(img_X)
        img_Y = np.array(img_Y)/127.5 - 1.
        img_Y = np.expand_dims(img_Y, axis=0)

        return img_X, img_Y

    def get_train_images_generator(self, path, batch_size=1):

        path_X = glob(path + "/trainA/*.jpg")
        path_Y = glob(path + "/trainB/*.jpg")

        n_batches = int(min(len(path_X), len(path_Y)) / batch_size)
        total_samples = n_batches * batch_size

        path_X = np.random.choice(path_X, total_samples, replace=False)
        path_Y = np.random.choice(path_Y, total_samples, replace=False)

        for i in range(n_batches-1):
            batch_A = path_X[i*batch_size:(i+1)*batch_size]
            batch_B = path_Y[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.load_image(img_A)
                img_B = self.load_image(img_B)

                img_A = self.imresize(img_A, self.image_size)
                img_B = self.imresize(img_B, self.image_size)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def get_train_images(self, path):

        path_X = glob(path + "/trainA/*.jpg")
        path_Y = glob(path + "/trainB/*.jpg")

        imgs_A, imgs_B = [], []
        for img_A, img_B in zip(path_X, path_Y):
            img_A = self.load_image(img_A)
            img_B = self.load_image(img_B)

            img_A = self.imresize(img_A, self.image_size)
            img_B = self.imresize(img_B, self.image_size)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B


if __name__ == "__main__":

    import os

    image_size = (256, 256)
	
    #  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #
    #  StyleStransfer(
    #      output_channels=3,
    #      image_size=image_size,
    #      n_epochs=100
    #  ).train_batch(
    #      batch_generator=ImageLoader(image_size).get_train_images_generator,
    #      train_path='monet2photo',
    #      results_path='training_results'
    #  )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    StyleStransfer(
        output_channels=3,
        image_size=image_size,
        n_epochs=100
    ).train(
        data_loader=ImageLoader(image_size).get_train_images,
        train_path='monet2photo',
        model_name='style_transfer'
    )
