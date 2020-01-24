import os
import glob
import argparse
import numpy as np
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from layers import BilinearUpSampling2D
from utils import load_images, display_images
from model import create_model

# Argument Parser
parser = argparse.ArgumentParser(description='Predict depth for images in given path')
parser.add_argument('--model', default='model.h5', type=str, help='Trained Keras model file')
parser.add_argument('--input', default='examples HELM', type=str, help='Path to example images')
parser.add_argument('--scale', default=150, type=float, help='Scale of color map in meters')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = create_model(existing = args.model)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images(glob.glob(args.input + "/*.*"))
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = model.predict(inputs, 4)

# Display results
viz = display_images(outputs.copy(), inputs.copy(), scale = args.scale)
Image.fromarray((viz * 255).astype(np.uint8)).save("test.png")
