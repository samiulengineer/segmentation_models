import os 
import cv2
import glob
import json
import shutil
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
from patchify import patchify
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# unpack labels        
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#') 
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#') 
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#') 
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155




def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """


    label_seg = np.zeros(label.shape,dtype = np.uint8)
    label_seg [np.all(label == Building, axis = -1)] = 0
    label_seg [np.all(label == Land, axis = -1)] = 1
    label_seg [np.all(label == Road, axis = -1)] = 2
    label_seg [np.all(label == Vegetation, axis = -1)] = 3
    label_seg [np.all(label == Water, axis = -1)] = 4
    label_seg [np.all(label == Unlabeled, axis = -1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg




def read_img(directory, rgb=False, norm=False):
    """
    Summary:
        read image with opencv and normalize the feature
    Arguments:
        directory (str): image path to read
        rgb (bool): convert BGR to RGB image as opencv read in BGR format
    Return:
        numpy.array
    """


    if rgb:
        return cv2.cvtColor(cv2.imread(directory, 1), cv2.COLOR_BGR2RGB) # read and convert from BGR to RGB
    elif norm:
        return cv2.imread(directory, 1)/255 # MinMaxScaler can be used for normalize
    else:
        return cv2.imread(directory, 1) 




def move_images_mask_from_tile(config):
    """
    Summary:
        Pacify images and masks after read from Tile folders.
        Save each pacify images and masks into images and masks folder under dataset_dir
    Arguments:
        config (dict): Configuration directory
    Return:
        return directory of saved patchify images and masks.
    """


    # create MinMaxScaler object
    scaler = MinMaxScaler((0.0,0.9999999)) # sklearn sometime give value greater than 1
    pathlib.Path((config['dataset_dir']+'imgs')).mkdir(parents = True, exist_ok = True) # creating image directory under dataset directory
    img_patch_dir = config['dataset_dir']+'imgs/'

    for path, subdirs, files in os.walk(config['dataset_dir'], topdown=True):
        # print(sorted(subdirs))
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':   #Find all 'images' directories
            images = os.listdir(path)#List of all image names in this subdirectory
            images = sorted(images)
            for i, image_name in enumerate(images):  
                
                if image_name.endswith(".jpg"):   #Only read jpg images...
                
                    image = read_img((path+"/"+image_name), norm=False)  #Read each image as BGR
                    print(image.shape)
                    SIZE_X = (image.shape[1]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    print(image.size)
                    image = np.array(image)         
        
                    #Extract patches from each image
                    print("Now patchifying image:", path+"/"+image_name)
                    patches_img = patchify(image, (config['patch_size'], config['patch_size'], 3), step=config['patch_size'])  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:]
                            
                            #Use minmaxscaler instead of just dividing by 255.
                            scaler.fit(single_patch_img.reshape(-1, single_patch_img.shape[-1]))
                            single_patch_img = scaler.transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                            save_path = path.replace("\\images","").split("/")[-1]
                            save_path = img_patch_dir+save_path.replace(" ","_")+"_patch_{}_{}".format(i, j)+"_"+image_name
                            plt.imsave(save_path, single_patch_img)  

    pathlib.Path((config['dataset_dir']+'masks')).mkdir(parents = True, exist_ok = True) # creating masks directory under dataset directory
    mask_patch_dir = config['dataset_dir']+'masks/'
    for path, subdirs, files in os.walk(config['dataset_dir'], topdown=True):
            
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':   #Find all 'images' directories
            masks = os.listdir(path)  #List of all image names in this subdirectory
            masks = sorted(masks)
            for i, mask_name in enumerate(masks): 

                if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                
                    mask = read_img((path+"/"+mask_name), rgb=True)  #Read each image as Grey (or color but remember to map each color to an integer)
                    SIZE_X = (mask.shape[1]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//config['patch_size'])*config['patch_size'] #Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)             
        
                    #Extract patches from each image
                    print("Now patchifying mask:", path+"/"+mask_name)
                    patches_mask = patchify(mask, (config['patch_size'], config['patch_size'], 3), step = config['patch_size'])  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
                            save_path = path.replace("\\masks","").split("/")[-1]
                            save_path = mask_patch_dir+save_path.replace(" ","_")+"_patch_{}_{}".format(i, j)+"_"+mask_name
                            plt.imsave(save_path, single_patch_mask)
    return img_patch_dir, mask_patch_dir




def data_split(images, masks, config):
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
        config (dict): Configuration directory
    Return:
        return the split data.
    """


    x_train, x_rem, y_train, y_rem = train_test_split(images, masks, train_size = config['train_size'])
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)
    return x_train, y_train, x_valid, y_valid, x_test, y_test




def move_image(dir_name, directories):
    """
    Summary:
        create new directory and move files into
        new created directory
    Arguments:
        dir_name (str): directory to create
        directories (list): directory where file will move
    Output:
        directory of move image
    """


    pathlib.Path(dir_name).mkdir(parents = True, exist_ok = True)
    for i in range(len(directories)):
        shutil.move(directories[i], dir_name) # move directory[i] file to dir_name



def data_split_and_foldering(config):
    """
    Summary:
        split data and foldering using above helper functions
    Arguments:
        config (dict): Configuration directory
    Output:
        data split and save in folders
    """


    print("Start data split and foldering....")
    img_dir, mask_dir = move_images_mask_from_tile(config) # save pachify images and masks directory

    # sorting so that there is no mis-match between image and mask
    image_dataset = sorted(glob.glob((img_dir+'*.jpg')))
    mask_dataset = sorted(glob.glob((mask_dir+'*.png')))


    print("Total number of images : {}".format(len(image_dataset)))
    print("Total number of masks : {}".format(len(mask_dataset)))

    # split data
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(image_dataset, mask_dataset, config)

    # directory create and move img
    os.mkdir((config['dataset_dir']+'train'))
    move_image((config['dataset_dir']+'train/img'), x_train)
    move_image((config['dataset_dir']+'train/mask'), y_train)
    
    os.mkdir((config['dataset_dir']+'valid'))
    move_image((config['dataset_dir']+'valid/img'), x_valid)
    move_image((config['dataset_dir']+'valid/mask'), y_valid)
    
    os.mkdir((config['dataset_dir']+'test'))
    move_image((config['dataset_dir']+'test/img'), x_test)
    move_image((config['dataset_dir']+'test/mask'), y_test)

    os.rmdir((config['dataset_dir']+'imgs'))
    os.rmdir((config['dataset_dir']+'masks'))
    
    print("Complete data split and foldering.")



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


    return to_categorical(rgb_to_2D_label(label), num_classes = num_classes)

if __name__ == "__main__":
    
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    # if not (os.path.exists(config['train_dir'])):
    #     data_split_and_foldering(config)
    # else:
    #     print("Loading image and mask directories.....")
    
    x_train = sorted(glob.glob(((config['dataset_dir']+'train/img')+'/*.jpg')))
    x_valid = sorted(glob.glob(((config['dataset_dir']+'valid/img')+'/*.jpg')))
    x_test = sorted(glob.glob(((config['dataset_dir']+'test/img')+'/*.jpg')))
    y_train = sorted(glob.glob(((config['dataset_dir']+'train/mask')+'/*.png')))
    y_valid = sorted(glob.glob(((config['dataset_dir']+'valid/mask')+'/*.png')))
    y_test = sorted(glob.glob(((config['dataset_dir']+'test/mask')+'/*.png')))

    train = pd.DataFrame.from_dict({"feature_ids": x_train, "masks": y_train})
    valid = pd.DataFrame.from_dict({"feature_ids": x_valid, "masks": y_valid})
    test = pd.DataFrame.from_dict({"feature_ids": x_test, "masks": y_test})

    train.to_csv((config["dataset_dir"]+"train.csv"), index=False)
    valid.to_csv((config["dataset_dir"]+"valid.csv"), index=False)
    test.to_csv((config["dataset_dir"]+"test.csv"), index=False)

