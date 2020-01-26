
from dense_depth.helper import Estimator
from matplotlib import pyplot
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of dense_depth package')
    parser.add_argument('--model', default='model.h5', type=str, help='Trained Keras model file')
    parser.add_argument('--input', default='image.jpg', type=str, help='Path to image')
    args = parser.parse_args()

    estimator = Estimator(args.model)

    depth, error = estimator.predict(args.input)

    pyplot.imshow(depth)

    pyplot.show()