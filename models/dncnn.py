from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Conv2D, Activation, BatchNormalization



"""
DNCNN Model
Paper: https://ieeexplore.ieee.org/abstract/document/7839189
# ----------------------------------------------------------------------------------------------
"""
def DnCNN(config):
    
    """
        Summary:
            Create DNCNN model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inpt = Input(shape=(config['height'], config['width'], config['in_channels']))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(config['num_classes'], (1, 1), activation='softmax',dtype='float32')(x)
    # x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model