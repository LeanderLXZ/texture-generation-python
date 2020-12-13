import time
import os
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from random import randint
from math import floor
from skimage import io
from PIL import Image

class TextureSynthesis(object):
    
    def __init__(self,
                 file_name,
                 texture_path,
                 save_path,
                 input_size,
                 window_size,
                 output_size,
                 apply_gaussian=True):
        
        self.file_name = file_name
        self.texture_path = texture_path
        self.save_path = save_path
        self.input_size = input_size
        self.window_size = window_size
        for i in range(2):
            if self.window_size[i] % 2 == 0:
                self.window_size[i] += 1
        self.output_size = output_size
        self.apply_gaussian = apply_gaussian
        self.texture_img = self.load_texture_img()
    
    def imresize(self, image, size):
        im = Image.fromarray(image)
        return np.array(im.resize(size, Image.BICUBIC))
    
    def load_texture_img(self):
        print('Loading image: ', self.texture_path)
        texture_img = io.imread(self.texture_path)
        texture_img = self.imresize(texture_img, self.input_size)

        img = Image.fromarray(np.uint8(texture_img))
        os.makedirs(self.save_path, exist_ok=True)
        img.save(os.path.join(self.save_path, self.file_name[:-4] + '_0.jpg'))

        texture_img = texture_img / 255.0
        
        # Remove Channels if # of channels > 3
        if (np.shape(texture_img)[-1] > 3):
            texture_img = texture_img[:,:,:3]
        # Convert Grayscale to RGB
        elif (len(np.shape(texture_img)) == 2):
            texture_img = np.repeat(texture_img[np.newaxis, :, :], 3, axis=0)
        
        return texture_img
    
    def generate_canvas(self):
        # Get texture_img dimensions
        n_rows, n_cols, n_channels = np.shape(self.texture_img)
        
        #create empty canvas
        canvas = np.zeros((*self.output_size, n_channels))
        # The map showing which pixels have been resolved
        filled_map = np.zeros(self.output_size)
        
        # Initialize a random 3x3 block
        margin = 1
        rand_row = randint(margin, n_rows - margin - 1)
        rand_col = randint(margin, n_cols - margin - 1)
        texture_img_patch = self.texture_img[
            rand_row-margin:rand_row+margin+1,
            rand_col-margin:rand_col+margin+1]
        
        #put it in the center of our canvas
        center_row = floor(self.output_size[0] / 2)
        center_col = floor(self.output_size[1] / 2)
        canvas[center_row-margin:center_row+margin+1,
               center_col-margin:center_col+margin+1] = texture_img_patch
        # Mark the resolved pixels
        filled_map[center_row-margin:center_row+margin+1,
                   center_col-margin:center_col+margin+1] = 1
        return canvas, filled_map
    
    def distances_to_probability(self, distances):
        truncation = 0.8
        attenuation = 80
        probabilities = 1 - distances / np.max(distances)
        probabilities *= (probabilities > truncation)
        # Attenuate the values
        probabilities = pow(probabilities, attenuation)
        if np.sum(probabilities) == 0:
            probabilities = 1 - distances / np.max(distances)
            # Truncate the values
            probabilities *= \
                (probabilities > truncation * np.max(probabilities))
            probabilities = pow(probabilities, attenuation)
        # Normalization
        probabilities /= np.sum(probabilities)
        return probabilities

    def get_best_candidate_coordinates(self, candidate_map):
        candidate_row = floor(np.argmax(candidate_map) / self.output_size[0])
        candidate_col = \
            np.argmax(candidate_map) - candidate_row * self.output_size[1]
        return candidate_row, candidate_col

    def get_neighbor(self, input_map, row, col):
        half_size = \
            (floor(self.window_size[0] / 2), floor(self.window_size[1] / 2))
        
        if input_map.ndim == 3:
            n_pad = ((half_size[0], half_size[0]), 
                     (half_size[1], half_size[1]), 
                     (0, 0))
        elif input_map.ndim == 2:
            n_pad = ((half_size[0], half_size[0]), 
                     (half_size[1], half_size[1]))
        else:
            raise 'ERROR: received a map of invalid dimension!'
        padded_map = np.lib.pad(input_map, n_pad, 'constant', constant_values=0)
        
        shifted_row = row + half_size[0]
        shifted_col = col + half_size[1]
        row_start = shifted_row - half_size[0]
        row_end = shifted_row + half_size[0] + 1
        col_start = shifted_col - half_size[1]
        col_end = shifted_col + half_size[1] + 1
        
        return padded_map[row_start:row_end, col_start:col_end]

    def update_candidate_map(self, candidate_map, filled_map):
        candidate_map *= 1 - filled_map #remove resolved from the map
        if np.argmax(candidate_map) == 0:
            for r in range(np.shape(candidate_map)[0]):
                for c in range(np.shape(candidate_map)[1]):
                    candidate_map[r, c] = \
                        np.sum(self.get_neighbor(filled_map, r, c))

    def get_texture_patches(self):
        n_rows, n_cols, n_channels = np.shape(self.texture_img)
        
        #find out possible steps for a search window to slide along the image
        num_horiz_patches = n_rows - (self.window_size[0] - 1)
        num_vert_patches = n_cols - (self.window_size[1] - 1)
        
        #init candidates array
        texture_patches = np.zeros((num_horiz_patches*num_vert_patches,
                                    self.window_size[0],
                                    self.window_size[1],
                                    n_channels))
        
        #populate the array
        for r in range(num_horiz_patches):
            for c in range(num_vert_patches):
                texture_patches[r*num_vert_patches + c] = \
                    self.texture_img[r:r+self.window_size[0],
                                     c:c+self.window_size[1]]
                
        return texture_patches

    def gaussian_kernel(self, kern_x, kern_y, sigma=3):
        # X
        interval = (2*sigma+1.)/(kern_x)
        x = np.linspace(-sigma-interval/2., sigma+interval/2., kern_x+1)
        kern1d_x = np.diff(st.norm.cdf(x))

        # Y
        interval = (2*sigma+1.)/(kern_y)
        x = np.linspace(-sigma-interval/2., sigma+interval/2., kern_y+1)
        kern1d_y = np.diff(st.norm.cdf(x))
        
        kernel_raw = np.sqrt(np.outer(kern1d_x, kern1d_y))
        return kernel_raw/kernel_raw.sum()
    
    def synthesis(self):
        
        start_time = time.time()
        
        # Initialize the image to be generated
        canvas, filled_map = self.generate_canvas()
        
        # Precalculate the array of examples patches from the texture image
        texture_patches = self.get_texture_patches()
        
        # Initialize a map for best candidates to be resolved
        candidate_map = np.zeros(np.shape(filled_map))
        for _ in tqdm(range(9, self.output_size[0] * self.output_size[1])):
            
            # Update the candidate map
            self.update_candidate_map(candidate_map, filled_map)

            # Get best candidate coordinates
            candidate_row, candidate_col = \
                self.get_best_candidate_coordinates(candidate_map)

            # Get a candidate_patch to compare to
            candidate_patch = self.get_neighbor(
                canvas, candidate_row, candidate_col)

            # Get a mask map
            candidate_patch_mask = self.get_neighbor(
                filled_map, candidate_row, candidate_col)
            # Applied gaussian kernel
            if self.apply_gaussian:
                candidate_patch_mask *= self.gaussian_kernel(
                    np.shape(candidate_patch_mask)[0],
                    np.shape(candidate_patch_mask)[1])
            # Cast to 3d array
            candidate_patch_mask = \
                np.repeat(candidate_patch_mask[:, :, np.newaxis], 3, axis=2)

            # Compare with every texture patch and construct the distance metric
            # Copy everything to match the dimensions of patches
            texture_patches_num = np.shape(texture_patches)[0]
            candidate_patch_mask = \
                np.repeat(candidate_patch_mask[np.newaxis, :, :, :, ],
                          texture_patches_num, axis=0)
            candidate_patch = \
                np.repeat(candidate_patch[np.newaxis, :, :, :, ],
                          texture_patches_num, axis=0)
            distances = \
                candidate_patch_mask * pow(texture_patches - candidate_patch, 2)
            distances = \
                np.sum(np.sum(np.sum(distances, axis=3), axis=2), axis=1)
            # Convert distances to probabilities
            probabilities = self.distances_to_probability(distances)
            
            # Sample the constructed PMF and fetch the appropriate pixel value
            sample = np.random.choice(
                np.arange(texture_patches_num), 1, p=probabilities)
            chosen_patch = texture_patches[sample]
            chosen_pixel = np.copy(chosen_patch[
                0,
                floor(self.window_size[0] / 2),
                floor(self.window_size[1] / 2)])

            # Resolve pixel
            canvas[candidate_row, candidate_col, :] = chosen_pixel
            filled_map[candidate_row, candidate_col] = 1

        img = Image.fromarray(np.uint8(canvas*255))
        # save_file_name = '{}_{}x{}_{}_result.jpg'.format(
        #     self.file_name[:-4],
        #     self.window_size[0],
        #     self.window_size[1],
        #     1 if self.apply_gaussian else 0)
        save_file_name = self.file_name[:-4] + '_1.jpg'
        
        os.makedirs(self.save_path, exist_ok=True)
        img.save(os.path.join(self.save_path, save_file_name))
        print('Runtime of {}: {:.2f}'.format(
            save_file_name, time.time() - start_time))


if __name__ == '__main__':
    
    texture_paths = ['texture4.jpg', 'texture5.jpg', 'texture6.jpg']
    window_sizes = [[3, 3], [5, 5], [11, 11], [15, 15], [21, 21]]
    
    for texture_path in texture_paths:
        for window_size in window_sizes:
            TextureSynthesis(
                texture_path=texture_path,
                save_path='./results/',
                input_size=[50, 50],
                window_size=window_size,
                output_size=[100, 100],
                apply_gaussian=True
                ).synthesis()
