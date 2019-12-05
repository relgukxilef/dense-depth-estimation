import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data, display_images
from model import create_model
from data import get_diode_train_test_data
#from callbacks import get_nyu_callbacks

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as k

from matplotlib import pyplot as plt
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
train_generator, test_generator = get_diode_train_test_data(args.bs)

#example_color, example_depth = train_generator[0]
#viz = display_images(
#    0.6 / np.maximum(example_depth[:, :, :, :1], 1e-3) / 350 * 
#    example_depth[:, :, :, 1:],  
#    example_color.copy()
#)
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.show()
#exit()

# Create the model
model = create_model( existing=args.checkpoint )

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
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

# Multi-gpu setup:
basemodel = model

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
model.fit_generator(
    train_generator, 
    epochs=args.epochs, 
    shuffle=True
)

# Save the final trained model:
basemodel.save_weights('model.h5')
