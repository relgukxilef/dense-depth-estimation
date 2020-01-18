import os
import argparse

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from model import create_model
from loss import depth_loss_function, depth_variance, error_variance
from data import DiodeSequence, HELMSequence
import tensorflow as tf
import tensorflow.keras.backend as k

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='model.h5', type=str, help='Trained Keras model file.')
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    session = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    k.set_session(session)

    print('Loading model...')

    model = create_model(existing = args.model)
    model.compile(
        loss=depth_loss_function, 
        metrics=[
            #depth_variance, 
            error_variance
        ]
    )

    test_generator = DiodeSequence(4, True)
    test_generator = HELMSequence(4)

    print(model.evaluate_generator(
        test_generator, workers = 2, use_multiprocessing = True
    ))