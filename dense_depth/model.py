import sys

from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from dense_depth.layers import BilinearUpSampling2D

def create_model(existing=''):
    """Create Keras model
    This method is mostly unchanged from the original DenseDepth implementation.
    """
        
    print('Loading base model (DenseNet)..')

    # Encoder Layers
    base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False)

    print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True

    # Starting number of decoder filters
    decode_filters = int(int(base_model_output_shape[-1])/2)

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
        up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

    decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
    if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

    # Extract depths (final layer)
    conv3 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

    # Create the model
    model = Model(inputs=base_model.input, outputs=conv3)

    if existing != '':
        model.load_weights(existing)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model