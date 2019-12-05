import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data
from model import create_model
from data import get_diode_data_filenames
#from callbacks import get_nyu_callbacks

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as k

import tensorflow as tf
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=1, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=0.6, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=350.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_diode', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')

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
dataset = tf.data.Dataset.from_tensor_slices(get_diode_data_filenames())

def load_npy(filename):
    return np.load(filename)

def load_sample(filenames):
    color = tf.ensure_shape(tf.cast(
        tf.image.decode_png(tf.read_file(filenames[0]), 3), tf.float32
    ) / 255, [768, 1024, 3])
    depth = tf.ensure_shape(
        tf.numpy_function(load_npy, [filenames[1]], [tf.float32])[0], 
        [768, 1024, 1]
    )
    mask = tf.ensure_shape(
        tf.numpy_function(load_npy, [filenames[2]], [tf.float32])[0],
        [768, 1024]
    )

    mask = tf.expand_dims(mask, -1)

    sample = tf.concat([color, depth, mask], -1)
    tf.image.central_crop(sample, np.random.uniform(0.5, 1.0))
    tf.image.resize(sample, [768 // 2, 1024 // 2])
    
    color, depth_mask = tf.split(sample, [3, 2], -1)

    depth_mask = tf.image.resize(depth_mask, [768 // 4, 1024 // 4])

    depth, mask = tf.split(depth_mask, [1, 1], -1)

    depth /= tf.clip_by_value(mask, 1e-2, 1)
    depth = tf.clip_by_value(depth, 0.6, 350)

    depth_mask = tf.concat([depth, mask], -1)

    return color, depth_mask

dataset = dataset.map(load_sample, 1).repeat().batch(args.bs)

# Create the model
model = create_model( existing=args.checkpoint )

# Training session details
runID = str(int(time.time())) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if False:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
#callbacks = get_nyu_callbacks(model, basemodel, dataset, dataset, load_test_data() if args.full else None , runPath)

# Start training
model.fit(
    dataset,
    epochs=args.epochs,
    steps_per_epoch=100
)

# Save the final trained model:
basemodel.save_weights(runPath + '/model.h5')
