import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function, depth_variance, error_variance
from utils import display_images
from model import create_model
from data import DiodeSequence

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as k

import tensorflow as tf
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Train network with given parameters')
    parser.add_argument('--data', type=str, default="diode_data", help='Path to training data')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--bs', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
    parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model')

    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    session = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    k.set_session(session)

    # Inform about multi-gpu training
    if args.gpus == 1: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
        print('Will use GPU ' + args.gpuids)
    else:
        print('Will use ' + str(args.gpus) + ' GPUs.')

    # Data loaders
    train_generator = DiodeSequence(args.bs, args.data)

    # Create the model
    model = create_model(existing=args.checkpoint)

    # Optimizer
    optimizer = Adam(lr=args.lr, amsgrad=True)

    # Compile the model
    model.compile(
        loss=depth_loss_function, optimizer=optimizer, 
        metrics=[depth_variance, error_variance]
    )

    print('Ready for training!\n')

    # Start training
    model.fit_generator(
        train_generator, 
        epochs=args.epochs, 
        shuffle=True,
        workers=2,
        use_multiprocessing=True
    )

    # Save the final trained model:
    model.save_weights('model.h5')
