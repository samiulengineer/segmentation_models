import math
import numpy as np
import tensorflow as tf
import rasterio
from PIL import Image
from tensorflow.keras.utils import Sequence, to_categorical



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

def read_tif_file(directory):
    with rasterio.open(directory) as f:
        data = f.read(1)
    return data

def read_other_file(directory):
    data = Image.open(directory)
    return np.array(data)


def read_img(directory, label=False, patch_idx=None, img_suffix=None):
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
        img_suffix (str): image type
    Return:
        numpy.array
    """

    if img_suffix == "tif":
        data = read_tif_file(directory)
    else:
        data = read_other_file(directory)
    
    if patch_idx:
        if len(data.shape())>2:
            return data[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3],:] # extract patch from original features
        return data[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
    else:
        return data

# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, in_channels, 
                 batch_size, num_class, patchify,
                 transform_fn=None, augment=None, weights=None, patch_idx=None, img_read_fn=None, img_suffix=None):
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            in_channels (int): number of input channels
            batch_size (int): how many data to pass in a single step
            patchify (bool): set TRUE if patchify experiment
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imblance class
            patch_idx (list): list of patch indices
            img_read_fn (function): function for read images by default use rasterio for read .tif images and PIL for other format.
            img_suffix (str): image extension or type
        Return:
            class object
        """


        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_data if transform_fn==None else transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights
        self.read = read_img if img_read_fn == None else img_read_fn
        self.img_suffix = img_suffix



    def __len__(self):
        """
        return total number of batch to travel full dataset
        """


        return math.ceil(len(self.img_dir) / self.batch_size)


    def __getitem__(self, idx):
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """


        # get index for single batch
        batch_x = self.img_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        
        if self.patchify:
            batch_patch = self.patch_idx[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        imgs = []
        tgts = []
        for i in range(len(batch_x)):
            if self.patchify:
                imgs.append(self.read(batch_x[i], patch_idx=batch_patch[i]))
                # transform mask for model
                if self.transform_fn:
                    tgts.append(self.transform_fn(self.read(batch_y[i], label=True,patch_idx=batch_patch[i]), self.num_class))
                else:
                    tgts.append(self.read(batch_y[i], label=True,patch_idx=batch_patch[i]))
            else:
                imgs.append(self.read(batch_x[i]))
                # transform mask for model
                if self.transform_fn:
                    tgts.append(self.transform_fn(self.read(batch_y[i], label=True), self.num_class))
                else:
                    tgts.append(self.read(batch_y[i], label=True))
        
        # augment data using Augment class above if augment is true
        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir, self.patch_idx) # augment images and mask randomly
                imgs = imgs+aug_imgs
            else:
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir) # augment images and mask randomly
                imgs = imgs+aug_imgs

            # transform mask for model
            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts+aug_masks


        tgts = np.array(tgts)
        imgs = np.array(imgs)        

        if self.weights != None:
            
            class_weights = self.weights/tf.reduce_sum(self.weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(tgts, tf.int32))#([self.paths[i] for i in indexes])

            return tf.convert_to_tensor(imgs), y_weights

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)
    

    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """



        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        
        imgs = []
        tgts = []
        if self.patchify:
            imgs.append(self.read(self.img_dir[idx],patch_idx=self.patch_idx[idx]))
            
            # transform mask for model
            if self.transform_fn:
                tgts.append(self.transform_fn(self.read(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]), self.num_class))
            else:
                tgts.append(self.read(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]))
        else:
            imgs.append(self.read(self.img_dir[idx]))
            
            # transform mask for model
            if self.transform_fn:
                tgts.append(self.transform_fn(self.read(self.tgt_dir[idx], label=True), self.num_class))
            else:
                tgts.append(self.read(self.tgt_dir[idx], label=True))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx