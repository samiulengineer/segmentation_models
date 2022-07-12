import math
import numpy as np
import albumentations as A


# Data Augment class
# ----------------------------------------------------------------------------------------------
class Augment:
    def __init__(self, batch_size, channels, ratio=0.5, seed=42, img_read_fn=None):
        super().__init__()
        """
        Summary:
            initialize class variables
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
            img_read_fn (function): function for read image
        Return:
            class object
        """


        self.ratio=ratio
        self.channels= channels
        self.aug_img_batch = math.ceil(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),])
        self.read = img_read_fn

    def call(self, feature_dir, label_dir, patch_idx=None):
        """
        Summary:
            randomly select a directory and augment data 
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """

        # choose random image from dataset to augment
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []

        for i in aug_idx:
            if patch_idx:
                img = self.read(feature_dir[i], in_channels = self.channels, patch_idx=patch_idx[i])
                mask = self.read(label_dir[i], label=True,patch_idx=patch_idx[i])
            else:
                img = self.read(feature_dir[i], in_channels = self.channels)
                mask = self.read(label_dir[i], label=True)
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented['image'])
            labels.append(augmented['mask'])
        return features, labels
