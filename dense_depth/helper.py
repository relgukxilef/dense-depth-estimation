
import numpy as np
import skimage.transform
from dense_depth import model
from PIL import Image

class Estimator():
    """Wraps the Keras model to provide a more natural interface.
    """

    def __init__(self, checkpoint = "model.h5"):
        """Loads the model at given path for prediction.
        """

        # load model
        self.model = model.create_model(checkpoint)

    def predict(self, image):
        """Predict the depth and rmse for the image at the given path.
        """

        # load an image and scale it to range [0, 1]
        image = np.clip(np.asarray(Image.open(image), dtype=float) / 255, 0, 1)

        # scale to the size the model was trained on
        # other sizes might work too
        image = skimage.transform.resize(image, [384, 512])

        # run image through model
        outputs = self.model.predict(image[None, :, :, :])

        depth = outputs[0, :, :, 0]
        depth2 = outputs[0, :, :, 1]
        variance = np.maximum(depth2 - depth * depth, 0.0)

        return depth * 350, variance**0.5 * 350