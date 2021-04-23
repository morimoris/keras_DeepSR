from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Add, Conv2DTranspose, Concatenate

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

def DeepSR():
    input_0 = Input(shape = (None, None, 1), name = "input_0")
    input_1 = Input(shape = (None, None, 1), name = "input_1")
    input_2 = Input(shape = (None, None, 1), name = "input_2")
    input_3 = Input(shape = (None, None, 1), name = "input_3")

    input_shape = Concatenate()([input_0, input_1, input_2, input_3])

    conv2d_0 = Conv2D(filters = 256,
                    kernel_size = (11, 11),
                    padding = "same",
                    activation = "tanh",
                    # kernel_regularizer = keras.regularizers.l1(0.01),
                    )(input_shape)

    conv2d_1 = Conv2D(filters = 512,
                    kernel_size = (1, 1),
                    padding = "same",
                    activation = "tanh",
                    # kernel_regularizer = keras.regularizers.l1(0.01),
                    )(conv2d_0)

    conv2d_2 =  Conv2D(filters = 1,
                    kernel_size = (3, 3),
                    padding = "same",
                    activation = "tanh",
                    # kernel_regularizer = keras.regularizers.l1(0.01),
                    )(conv2d_1)

    deconv2d_0 = Conv2DTranspose(filters = 1,
                        kernel_size = (25, 25),
                        # kernel_regularizer = keras.regularizers.l1(0.01),
                        strides = (1, 1),
                        padding = "same")(conv2d_2)


    model = Model(inputs = [input_0, input_1, input_2, input_3], outputs = [deconv2d_0])

    model.summary()

    return model
