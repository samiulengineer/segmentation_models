import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import earthpy.plot as ep
import earthpy.spatial as es
from tensorflow import keras
from .augment import Augment
from datetime import datetime
import matplotlib.pyplot as plt
from .dataloader import MyDataset




# Prepare masks
# ----------------------------------------------------------------------------------------------
def create_mask(mask, pred_mask):
    """
    Summary:
        apply argmax on mask and pred_mask class dimension
    Arguments:
        mask (ndarray): image labels/ masks
        pred_mask (ndarray): prediction labels/ masks
    Return:
        return mask and pred_mask after argmax
    """
    mask = np.argmax(mask, axis = 3)
    pred_mask = np.argmax(pred_mask, axis = 3)
    return mask, pred_mask

# Sub-ploting and save
# ----------------------------------------------------------------------------------------------

def display(display_list, idx, directory, score, exp):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
        exp (str): experiment name
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if title[i]=="DEM":
            ax = plt.gca()
            hillshade = es.hillshade(display_list[title[i]], azimuth=180)
            ep.plot_bands(
                display_list[title[i]],
                cbar=False,
                cmap="terrain",
                title=title[i],
                ax=ax
            )
            ax.imshow(hillshade, cmap="Greys", alpha=0.5)
        elif title[i]=="VV" or title[i]=="VH":
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis('off')
        else:
            plt.title(title[i])
            plt.imshow((display_list[title[i]]))
            plt.axis('off')

    prediction_name = "img_ex_{}_{}_MeanIOU_{:.4f}.png".format(exp, idx, score) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


# Save all plot figures
# ----------------------------------------------------------------------------------------------
def show_predictions(dataset, model, config, val=False):
    """
    Summary: 
        save image/images with their mask, pred_mask and accuracy
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
        val (bool): for validation plot save
    Output:
        save predicted image/images
    """

    if val:
        directory = config['prediction_val_dir']
    else:
        directory = config['prediction_test_dir']

    # save single image after prediction from dataset
    if config['plot_single']:
        feature, mask, idx = dataset.get_random_data(config['index'])
        data = [(feature, mask)]
    else:
        data = dataset
        idx = 0

    for feature, mask in data: # save all image prediction in the dataset
        prediction = model.predict_on_batch(feature)
        mask, pred_mask = create_mask(mask, prediction)
        for i in range(len(feature)): # save single image prediction in the batch
            m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
            m.update_state(mask[i], pred_mask[i])
            score = m.result().numpy()
            display({"VV": feature[i][:,:,0],
                     "VH": feature[i][:,:,1],
                     "DEM": feature[i][:,:,2],
                      "Mask": mask[i],
                      "Prediction (MeanIOU_{:.4f})".format(score): pred_mask[i]
                      }, idx, directory, score, config['experiment'])
            idx += 1


# Create config path
# ----------------------------------------------------------------------------------------------
def create_paths(config):
    """
    Summary:
        parsing the config.yaml file and re organize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """
            
    config['height'] = config['patch_size']
    config['width'] = config['patch_size']
    
    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    
    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    
    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_epochs_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+config['model_name']+'/'

    config['csv_log_name'] = "{}_ex_{}_epochs_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/csv_logger/'+config['model_name']+'/'

    config['checkpoint_name'] = "{}_ex_{}_epochs_{}_{}.hdf5".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'
    
    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/validation/'

    pathlib.Path(config['prediction_test_dir']).mkdir(parents = True, exist_ok = True)
    pathlib.Path(config['csv_log_dir']).mkdir(parents = True, exist_ok = True)
    pathlib.Path(config['tensorboard_log_dir']).mkdir(parents = True, exist_ok = True)
    pathlib.Path(config['checkpoint_dir']).mkdir(parents = True, exist_ok = True)
    pathlib.Path(config['prediction_val_dir']).mkdir(parents = True, exist_ok = True)

    return config


def get_train_val_dataloader(config):
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """
    config = create_paths(config)

    print("Loading features and masks directories.....")
    train_dir = pd.read_csv(config['train_dir'])
    valid_dir = pd.read_csv(config['valid_dir'])
    train_features = train_dir.feature_ids.values
    train_masks = train_dir.masks.values
    valid_features = valid_dir.feature_ids.values
    valid_masks = valid_dir.masks.values

    if config['patchify']:
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
    else:
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_features)))
    print("valid Example : {}".format(len(valid_features)))


    # create Augment object if augment is true
    if config['augment'] and config['batch_size']>1:
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    # class weight
    if config['weights']:
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None
    
    # create dataloader object
    train_dataset = MyDataset(train_features, train_masks,
                                in_channels=config['in_channels'], patchify=config['patchify'],
                                batch_size=n_batch_size, transform_fn=config["transform_data"], 
                                num_class=config['num_classes'], augment=augment_obj, 
                                weights=weights, patch_idx=train_idx, img_read_fn=config["img_read_fn"],
                                img_suffix=config["img_suffix"])

    val_dataset = MyDataset(valid_features, valid_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=config["transform_data"], 
                            num_class=config['num_classes'],patch_idx=valid_idx, img_read_fn=config["img_read_fn"],
                            img_suffix=config["img_suffix"])
    
    return train_dataset, val_dataset, config

def get_test_dataloader(config):
    """
    Summary:
        read test image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        test dataloader
    """

    print("Loading features and masks directories.....")
    test_dir = pd.read_csv(config['test_dir'])
    test_features = test_dir.feature_ids.values
    test_masks = test_dir.masks.values

    if config['patchify']:
        test_idx = test_dir['patch_idx']
    else:
        test_idx = None

    print("test Example : {}".format(len(test_features)))


    test_dataset = MyDataset(test_features, test_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=config["transform_data"], 
                            num_class=config['num_classes'],patch_idx=test_idx, img_read_fn=config["img_read_fn"],
                            img_suffix=config["img_suffix"])
    
    return test_dataset