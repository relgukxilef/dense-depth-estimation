
from dense_depth.helper import Estimator
from matplotlib import pyplot

estimator = Estimator("model.h5")

depth, error = estimator.predict("image.jpg")

pyplot.imshow(depth)

pyplot.show()