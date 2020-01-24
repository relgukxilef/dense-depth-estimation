import numpy as np
from PIL import Image
from skimage.transform import resize

def load_images(image_files):
    """Load each image in array of filenames and store them in a single tensor.
    """

    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ), dtype=float) / 255, 0, 1)
        x = resize(x, [768 // 2, 1024 // 2])
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def to_multichannel(i):
    """Turn grayscale images into color images, keep color images as they are.
    """

    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)
        
def display_images(outputs, inputs=None, gt=None, is_colormap=True, scale=150):
    """Visualize in- and outputs for testing.
    """

    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('rainbow')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(plasma(x))

        depth = np.minimum(outputs[i][:,:,0] * 350 / scale, 1.0)

        imgs.append(
            plasma(depth)[:,:,:3]
        )

        standard_deviation = np.maximum(
            outputs[i][:,:,1] - outputs[i][:,:,0] * outputs[i][:,:,0], 0
        ) ** 0.5 * 350 / scale
        imgs.append(
            plasma(np.minimum(standard_deviation, 1.0))[:,:,:3]
        )

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))
