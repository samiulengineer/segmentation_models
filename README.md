# segmentation_models

```Tensorflow.keras``` Implementation

## Models

the segmentation_models.models contains following models implementation.

| Model | Name | Reference |
|:---------------|:----------------|:----------------|
| `dncnn`     | DN-CNN         | [Zhang et al. (2017)](https://ieeexplore.ieee.org/document/7839189) |
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `vnet`      | V-net | [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) |
| `unet++` | U-net++         | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `u2net`     | U^2-Net         | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
| `fapnet`     | FAP-NET         | [In proceeding](#) |
| `attentionunet`  | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
| `linknet`     | LINK-Net         | [Chaurasia et al. (2017)](https://arxiv.org/pdf/1707.03718.pdf) |
| `dlinknet`     | DLINK-Net         | [Zhou et al. (2018)](hhttps://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf) |
| `deeplabv3`     | DeepLabV3         | [Chen et al. (2017)](https://arxiv.org/abs/1706.05587v3) |
| `deepwatermap`     | DeepWaterMAP         | [Isikdogan et al. (2020)](https://ieeexplore.ieee.org/document/8913594) |
| `dilatednet`     | DialatedNET         | [Yu et al. (2016)](https://arxiv.org/abs/1511.07122) |
| `enet`     | E-NET         | [Paszke et al. (2016)](https://arxiv.org/abs/1606.02147) |
| `segmenter`     | SEGMENTER         | [Strudel et al. (2021)](https://arxiv.org/abs/2105.05633) |

## Setup

First clone the github repo in your local or server machine by following:
```
git clone https://github.com/samiulengineer/segmentation_models.git
cd segmentation_models
```
**Note:** Remember to change directory.

Create a new conda environment and install dependency from `requirement.txt` file.

```
conda create --name <env> --file requirements.txt
```

## Training

To train a model follow the instruction bellow.

```
from utils.util import get_train_val_dataloader
from utils.callbacks import SelectCallbacks
from utils.loss import focal_loss
from models.unet import unet

# DataLoader
train, val, config = get_train_val_dataloader(config)

# Model
model = unet(config)

# Optimizer
adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])

# Loggers for training
loggers = SelectCallbacks(val, model, config)

# Compile and train
model.compile(optimizer = adam, loss = focal_loss())
model.fit(train,verbose = 1,epochs = 1,validation_data = val,shuffle = False, callbacks = loggers.get_callbacks())
```

## Example

Jupyter Notebokk provided as example.