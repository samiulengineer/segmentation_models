from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Conv2D, add, PReLU, concatenate, Conv2DTranspose, Dropout, BatchNormalization



"""
VNET Model
Paper: https://ieeexplore.ieee.org/abstract/document/7785132
# ----------------------------------------------------------------------------------------------
"""
def resBlock(input, stage, keep_prob, stage_num = 5):
    
    for _ in range(3 if stage>3 else stage):
        conv = PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input)))
        # print('conv_down_stage_%d:' %stage,conv.get_shape().as_list())
    conv_add = PReLU()(add([input, conv]))
    # print('conv_add:',conv_add.get_shape().as_list())
    conv_drop = Dropout(keep_prob)(conv_add)
    
    if stage < stage_num:
        conv_downsample = PReLU()(BatchNormalization()(Conv2D(16*(2**stage), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop)))
        return conv_downsample, conv_add
    else:
        return conv_add, conv_add
    
def up_resBlock(forward_conv,input_conv,stage):
    
    conv = concatenate([forward_conv, input_conv], axis = -1)
    
    for _ in range(3 if stage>3 else stage):
        conv = PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
        conv_add = PReLU()(add([input_conv,conv]))

    if stage > 1:
        conv_upsample = PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(stage-2)),2,strides = (2, 2),padding = 'valid',activation = None,kernel_initializer = 'he_normal')(conv_add)))
        return conv_upsample
    else:
        return conv_add
    
def vnet(config):
    
    """
        Summary:
            Create VNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    keep_prob = 0.99
    features = []
    stage_num = 5 # number of blocks
    input = Input((config['height'], config['width'], config['in_channels']))
    x = PReLU()(BatchNormalization()(Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input)))
    
    for s in range(1, stage_num+1):
        x, feature = resBlock(x, s, keep_prob, stage_num)
        features.append(feature)
        
    conv_up = PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(s-2)),2, strides = (2, 2), padding = 'valid', activation = None, kernel_initializer = 'he_normal')(x)))
    
    for d in range(stage_num-1, 0, -1):
        conv_up = up_resBlock(features[d-1], conv_up, d)

    output = Conv2D(config['num_classes'], 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
        
    model = Model(inputs = [input], outputs = [output])

    return model
