import numpy as np
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
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
        self.depth_size = [768 // 4, 1024 // 4, 1]

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

            color[i] = resize(np.clip(np.asarray(
                Image.open(sample[0])
            ) / 255, 0, 1), self.target_size)
            depth[i, :, :, :1] = self.minDepth / resize(
                np.clip(np.load(sample[1]), self.minDepth, self.maxDepth),
                self.depth_size
            )
            depth[i, :, :, 1] = np.clip(
                resize(np.load(sample[2]), self.depth_size[:2]), 
                0, 1
            )

        return color, depth