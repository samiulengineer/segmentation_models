from tensorflow.keras.layers import Conv2D, Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model


"""
FAPNET Model
Paper:
# ----------------------------------------------------------------------------------------------
"""

def fapnet(config):
    
    """
        Summary:
            Create MNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inputs = Input((config['height'], config['width'], config['in_channels']))
    
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)  # Original 0.1
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
     
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    p6 = MaxPooling2D((2, 2))(c6)
     
    c7 = Conv2D(1012, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Expansive path 
    
    u8 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c6])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c5])
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c4])
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.2)(c10)
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
     
    u11 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c3])
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.2)(c11)
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
     
    u12 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c2])
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
    c12 = Dropout(0.2)(c12)  # Original 0.1
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
     
    u13 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c12)
    u13 = concatenate([u13, c1], axis=3)
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
    c13 = Dropout(0.2)(c13)  # Original 0.1
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
     
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(c13)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
