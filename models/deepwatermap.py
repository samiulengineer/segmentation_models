import tensorflow as tf

"""
DEEPWATERMAP Model
Paper: https://ieeexplore.ieee.org/document/8913594
# ----------------------------------------------------------------------------------------------
"""

def deepwatermap(config, min_width=4):
    inputs = tf.keras.layers.Input(shape=(config['height'], config['width'], config['in_channels']))

    def conv_block(x, num_filters, kernel_size, stride=1, use_relu=True):
        x = tf.keras.layers.Conv2D(
                        filters=num_filters,
                        kernel_size=kernel_size,
                        kernel_initializer='he_uniform',
                        strides=stride,
                        padding='same',
                        use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if use_relu:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    def downscaling_unit(x):
        num_filters = int(x.get_shape()[-1]) * 4
        x_1 = conv_block(x, num_filters, kernel_size=5, stride=2)
        x_2 = conv_block(x_1, num_filters, kernel_size=3, stride=1)
        x = tf.keras.layers.Add()([x_1, x_2])
        return x

    def upscaling_unit(x):
        num_filters = int(x.get_shape()[-1]) // 4
        x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        x_1 = conv_block(x, num_filters, kernel_size=3)
        x_2 = conv_block(x_1, num_filters, kernel_size=3)
        x = tf.keras.layers.Add()([x_1, x_2])
        return x

    def bottleneck_unit(x):
        num_filters = int(x.get_shape()[-1])
        x_1 = conv_block(x, num_filters, kernel_size=3)
        x_2 = conv_block(x_1, num_filters, kernel_size=3)
        x = tf.keras.layers.Add()([x_1, x_2])
        return x

    # model flow
    skip_connections = []
    num_filters = min_width

    # first layer
    x = conv_block(inputs, num_filters, kernel_size=1, use_relu=False)
    skip_connections.append(x)

    # encoder
    for i in range(4):
        x = downscaling_unit(x)
        skip_connections.append(x)

    # bottleneck
    x = bottleneck_unit(x)

    # decoder
    for i in range(4):
        x = tf.keras.layers.Add()([x, skip_connections.pop()])
        x = upscaling_unit(x)

    # last layer
    x = tf.keras.layers.Add()([x, skip_connections.pop()])
    x = conv_block(x, config['num_classes'], kernel_size=1, use_relu=False)
    x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model