import numpy as np
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
import os
from sklearn.utils import shuffle
from skimage.transform import resize

def get_diode_train_test_data(batch_size):
    # TODO: train and test set shouldn't be the same
    data = DiodeSequence(batch_size)
    return data, data

class DiodeSequence(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = []

        self.image_size = [768, 1024, 3]
        self.target_size = [768 // 2, 1024 // 2, 3]
        self.depth_mask_size = [768 // 4, 1024 // 4, 2]

        self.policy = BasicPolicy(
            color_change_ratio=0.50, mirror_ratio=0.50, 
            flip_ratio=0.0, add_noise_peak=0, erase_ratio=-1.0
        )
        self.batch_size = batch_size
        self.shape_rgb = [batch_size] + self.target_size
        self.shape_depth = [batch_size] + [768 // 4, 1024 // 4, 2]
        self.maxDepth = 350.0
        self.minDepth = 0.6
        
        for root, _, files in os.walk("diode_data"):
            for file in files:
                if file.endswith(".png"):
                    image = os.path.join(root, file)
                    depth = image[:-4] + "_depth.npy"
                    mask = image[:-4] + "_depth_mask.npy"
                    self.dataset += [(image, depth, mask)]
                    
        self.dataset = shuffle(self.dataset, random_state=0)
        self.size = len(self.dataset)

    def __len__(self):
        return -(-self.size // self.batch_size) # division rounding up

    def __getitem__(self, idx, is_apply_policy=True):
        color = np.zeros(self.shape_rgb)
        depth = np.zeros(self.shape_depth)

        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.size - 1)

            sample = self.dataset[index]

            sample = np.concatenate(
                (
                    Image.open(sample[0]), 
                    np.load(sample[1]), 
                    np.load(sample[2])[:, :, None]
                ), -1
            )

            # center crop
            zoom = np.random.uniform(0.5, 1.0)
            size = [int(768 * zoom), int(1024 * zoom)]
            start = [768 // 2 - size[0] // 2, 1024 // 2 - size[1] // 2]
            end = [start[i] + size[i] for i in range(2)]
            sample = sample[start[0]:end[0], start[1]:end[1], :]

            # normalize depth
            sample[:, :, 3] = self.minDepth / np.clip(
                sample[:, :, 3], self.minDepth, self.maxDepth
            )
            sample[:, :, 3] *= sample[:, :, 4]

            # scale
            color[i] = resize(sample[:, :, :3], self.target_size[:2]) / 255
            depth[i] = resize(sample[:, :, 3:5], self.depth_mask_size[:2])

            # 80% chance of grayscale
            if np.random.uniform() < 0.8:
                color[i] = np.dot(color[i], [0.299, 0.587, 0.144])[:, :, None]

            # bad exposure
            clip_from = np.random.uniform(0, 0.2)
            clip_to = np.random.uniform(0.8, 1.0)
            spread_from = np.random.uniform(0, 0.4)
            spread_to = np.random.uniform(0.6, 1.0)
            
            color[i] = np.clip(
                (color[i] - clip_from) / (clip_to - clip_from), 0, 1
            ) * (spread_to - spread_from) + spread_from


        # filter out mixtures with invalid samples from resizing
        #depth[:, :, :, 0] /= np.maximum(depth[:, :, :, 1], 1e-3)
        depth[:, :, :, 1] = np.where(depth[:, :, :, 1] > 0.99, 1.0, 0.0)

        return color, depth