import os
import math
from tensorflow import keras
from .util import show_predictions


# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model, config):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        if (epoch % self.config['val_plot_epoch'] == 0): # every after certain epochs the model will predict mask
            show_predictions(self.val_dataset, self.model, self.config, True)

    def get_callbacks(self):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only = True))
        
        if self.config['tensorboard']:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir = os.path.join(self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))
        
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        if self.config['early_stop']:
            self.callbacks.append(keras.callbacks.EarlyStopping(monitor = 'my_mean_iou', patience = self.config['patience']))
        
        if self.config['val_pred_plot']:
            self.callbacks.append(SelectCallbacks(self.val_dataset, self.model, self.config))
        
        return self.callbacks