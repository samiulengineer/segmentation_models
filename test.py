import rasterio
import numpy as np
from tensorflow.keras.utils import to_categorical
from models.unet import unet
from tensorflow import keras
import tensorflow as tf
from utils.util import get_train_val_dataloader
from utils.callbacks import SelectCallbacks
from utils.loss import focal_loss
from utils.metrics import get_metrics

# labels normalization values       
label_norm = {0:["_vv.tif", -17.54, 5.15],
                1:["_vh.tif",-10.68, 4.62],
                2:["_nasadem.tif",166.47, 178.47],
                3:["_jrc-gsw-change.tif", 238.76, 5.15],
                4:["_jrc-gsw-extent.tif", 2.15, 22.71],
                5:["_jrc-gsw-occurrence.tif", 6.50, 29.06],
                6:["_jrc-gsw-recurrence.tif", 10.04, 33.21],
                7:["_jrc-gsw-seasonality.tif", 2.60, 22.79],
                8:["_jrc-gsw-transitions.tif", 0.55, 1.94]}


def transform_data(label, num_classes):
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """


    return to_categorical(label, num_classes = num_classes)



def read_img(directory, in_channels=None, label=False, patch_idx=None):
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        in_channels (bool): number of channels to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
    Return:
        numpy.array
    """

    if label:
        with rasterio.open(directory) as fmask:
            mask = fmask.read(1)
            mask[mask == 255] = 0 # convert unlabeled data
            if patch_idx:
                return mask[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]] # extract patch from original mask
            else:
                return mask
    else:
        X = np.zeros((512,512, in_channels))
        
        # read N number of channels
        for i in range(in_channels):
            tmp_ext = label_norm[i][0]
            with rasterio.open((directory+tmp_ext)) as f:
                fea = f.read(1)
            
            # normalize data
            X[:,:,i] = (fea - label_norm[i][1]) / label_norm[i][2]
        if patch_idx:
            return X[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3],:] # extract patch from original features
        else:
            return X

config = {# Image Input/Output
    # ----------------------------------------------------------------------------------------------
    "in_channels" : 3,
    "num_classes" : 2,


    # Training
    # ----------------------------------------------------------------------------------------------
    # unet/mod-unet/dncnn/u2net/vnet/unet++
    "model_name" : "mnet",
    "batch_size" : 3,
    "epochs" : 2,
    "learning_rate" : 3e-4,
    "val_plot_epoch" : 20,
    "augment" : False,
    "transfer_lr" : False,
    "gpu" : "0",

    # Regular/Cls_balance/Patchify/Patchify_WOC
    "experiment" : "cls_balance",

    # Patchify
    # ----------------------------------------------------------------------------------------------
    "patchify" : False,
    "patch_class_balance" : False, # whether to use class balance while doing patchify
    "patch_size" : 512, # height = width, anyone is suitable
    "stride" : 64,
    "p_train_dir" : "train_patch_256.json",
    "p_valid_dir" : "valid_patch_256.json",
    "p_test_dir" : "test_patch_256.json",

    # Dataset
    # ----------------------------------------------------------------------------------------------
    "weights" : True,
    "balance_weights" : [0.14, 0.86],
    "dataset_dir" : "D:/CSML_workPlace/flood_water_ditection_satellite/data/",
    "root_dir" : "D:/CSML_workPlace/test",
    "train_size" : 0.8,  # validation 10% and test 10%
    "train_dir" : "train.csv",
    "valid_dir" : "valid.csv",
    "test_dir" : "test.csv",

    # Logger/Callbacks
    # ----------------------------------------------------------------------------------------------
    "csv" : True,
    "val_pred_plot" : True,
    "lr" : True,
    "tensorboard" : True,
    "early_stop" : False,
    "checkpoint" : True,
    "patience" : 300, # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

    # Evaluation
    # ----------------------------------------------------------------------------------------------
    "load_model_name" : "m.hd5y",
    "load_model_dir" : None,  # if by default model directory change. Then change it here


    "plot_single" : True, # if True, then only index x_test image will plot
    "index" : -1, # by default -1 means random image else specific index image provide by user
    "transform_data": transform_data,
    "img_read_fn": read_img,
    "img_suffix": ".tif"
}

train, val, config = get_train_val_dataloader(config)
metrics = list(get_metrics(config).values())

model = unet(config)
adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])
loggers = SelectCallbacks(val, model, config)
model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)

model.fit(train,verbose = 1,epochs = 1,validation_data = val, 
            shuffle = False, callbacks = loggers.get_callbacks())
