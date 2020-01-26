# Dense Depth Estimation for Historic Images

This project consists of a tool to generate dense depth maps from images, specialized on historic imgae material. Additionally it consists of the evaluation of said tool and estimation of the accuracy of the results.

The Develop folder has scripts for training, testing and avaluating the model.

## Requirements

* Python 3.4 - 3.7. __Does not work with Python 3.8__ due to a limitation in TensorFlow.
* GPU with CUDA support.
* 4GB of VRAM for training with batch size 2. More for larger batches.

## Installation

Run `pip install .` in the root folder. That's it.

## Demo

The script `Demo/demo.py` shows how to predict the depth of an individual image. If it is run outside of the Demo directory it is ncessary to specify the path to the model and the test image with the parameters `--model` and `--input` respectively.

## Training

`Develop/train.py --bs 2 --epochs 5 --lr 1e-5 --data DIODE --checkpoint model.h5`

The script is able to load data from the DIODE dataset and train the model on it. The path to the folder containing the uncompressed image, depth and mask files must be specified using the `--data` parameter. 
The batch size, number of epochs and learning rate may be specified using the `--bs`, `--epochs` and `--lr` parameters respectively. Training can be continued from an existing checkpoint using the `--checkpoint` parameter.

## Testing

`Develop/test.py --model model.h5 --input examples --scale 150`

Creates a visualization of the model output using the images in the folder specified by `--input`. The model is specified by the `--model` parameter. The `--scale` specifies the maximum depth used for the color coding. The visualization is stored as an image called `test.png`.

## Evaluation

`Develop/evaluate.py --gt --path diode_test --model model.h5`

Calculates the real RMSE and the estimated RMSE using the model specified in the `--model` parameter. The folder containing all the test data is specified using the `--path` parameter. The real RMSE can only be calculated if the ground truth is present in the specified folder and is only calculated if `--gt` is present. It can be useful to specify a folder containing only images without ground truth to calculated the estimated RMSE.