"""
============================================
Lesson 1: Dogs vs Cats model
http://wiki.fast.ai/index.php/Lesson_1_Notes
============================================
"""

print(__doc__)

################################################################################
## Setup Vgg16 on Windows 8.1 Pro 64 Bit* and Python 3.5**: 
# *  Windows-8.1-6.3.9600-SP0
# ** Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) on Windows (64 bits)

## Get dependencies:
# ------------------------------------------------------------------------------------------------------------
# | Name      | Version                           | Source                                                   |
# ------------------------------------------------------------------------------------------------------------
# | h5py      | 2.7.0rc2-cp35-cp35m-win_amd64     | http://www.lfd.uci.edu/~gohlke/pythonlibs/#h5py          |
# | numpy     | 1.11.2                            | pip                                                      |
# | scipy     | 0.18.1                            | pip                                                      |
# | matplotlib| 1.5.3                             | pip                                                      |
# | bcolz     | 1.1.0-cp35-cp35m-win_amd64        | http://www.lfd.uci.edu/~gohlke/pythonlibs/#bcolz         |
# | theano    | 0.8.2                             | pip                                                      |
# | keras     | 1.2.0                             | pip                                                      |
# | nose      | 1.3.7                             | pip                                                      |
# | mingw     | x86_64-6.2.0-posix-seh-rt_v5-rev1 | https://sourceforge.net/projects/mingw-w64/              |
# | tdm-gcc   | tdm64-gcc-5.1.0-2                 | http://tdm-gcc.tdragon.net/download                      |
# | openblas  | v0.2.14-Win64-int32               | https://sourceforge.net/projects/openblas/files/v0.2.14/ |
# | mingw dll | v0.2.14-Win64                     | https://sourceforge.net/projects/openblas/files/v0.2.14/ |
# ------------------------------------------------------------------------------------------------------------
#
#
# Installation notes:
#
# ------------------------------------------------------------------------------------------------------------
# install tdm-gcc to C:\TDM-GCC-64
# install mingw to C:\mingw-w64
# install openblas to C:\openblas
# copy mingw dll files (libgcc_s_seh-1.dll, libgfortran-3.dll, libquadmath-0.dll) to C:\openblas\bin
# ------------------------------------------------------------------------------------------------------------

## Configure theano:
# Configuration file: C:\Users\[Current User]\.theanorc.txt
#
# ------------------------------------------------------------------------------------------------------------
# [global]
# device = cpu
# floatX = float32
# 
# [blas]
# ldflags = -LC:\\openblas\\bin -LC:\\openblas\\lib -lopenblas
# 
# [gcc] 
# cxxflags = -shared -I"C:\TDM-GCC-64\include" -I"C:\TDM-GCC-64\x86_64-w64-mingw32\include" -D_hypot=hypot
# ------------------------------------------------------------------------------------------------------------

## Configure keras:
# Configuration file: C:\Users\[Current User]\.keras\keras.json
#
# ------------------------------------------------------------------------------------------------------------
# {
#     "image_dim_ordering": "th",
#     "backend": "theano",
#     "floatx": "float32",
#     "epsilon": 1e-07
# }
# ------------------------------------------------------------------------------------------------------------

## Configure system variables:
# Configuration dialog: System -> Advanced system settings -> Environment Variables... -> System variables
# 
# ------------------------------------------------------------------------------------------------------------
# Variable name: PY_HOME
# Variable value: C:\Users\[Current User]\AppData\Local\Programs\Python\Python35
# ------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------
# Variable name: Path
# Variable value: %PY_HOME%;%PY_HOME%\Lib;%PY_HOME%\DLLs;%PY_HOME%\Lib\lib-tk;%PY_HOME%\Scripts;C:\mingw-w64\x86_64-6.2.0-posix-seh-rt_v5-rev1\mingw64\bin;c:\openblas\lib;c:\openblas\bin
# ------------------------------------------------------------------------------------------------------------

################################################################################
## Import libraries

import h5py
print("h5py " + h5py.__version__)

import theano
print("theano " + theano.__version__)

import nose
print("nose " + nose.__version__)

import matplotlib
from matplotlib import pyplot as plt
print("matplotlib " + matplotlib.__version__)

import numpy as np
print("numpy " + np.__version__)

import scipy
print("scipy " + scipy.__version__)

import keras
print("keras " + keras.__version__)

from importlib.machinery import SourceFileLoader
# http://wiki.fast.ai/index.php/Lesson_1_Notes
model = SourceFileLoader("Vgg16", "D:\\MOOCs\\fast.ai -- deep learning\\Lesson 1\\vgg16.py").load_module()
utils = SourceFileLoader("Utils", "D:\\MOOCs\\fast.ai -- deep learning\\Lesson 1\\utils.py").load_module()

import platform
print("system " + platform.platform())

################################################################################
## Get a pre-trained model to minimize computation time
# Detailed description of the vgg16 architecture:
# Original paper: https://arxiv.org/abs/1409.1556
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Check local config file: os.path.expanduser('~/.theanorc.txt')
# Print current configuration: print(theano.config)
# Change current configuration: theano.config.floatX = "float32"
# Unit testing: theano.test()  # requires nose

# Downloads per default a pre-trained model (vgg16_bn) of type h5 
# from http://www.platform.ai/models/ to C:\Users\[Current User]\.keras\models\vgg16_bn.h5
# vg16_bn uses batch normalization: https://arxiv.org/abs/1502.03167
vgg = model.Vgg16()


################################################################################
## Load training and validation image data

# Set param batch_size as large as you can, but no larger than 64 is recommended. 
# If you have an older or cheaper GPU, you'll run out of memory, eventually.
batch_size = 8

# The images must be in subdirectories named based on their category
path = "D:\\MOOCs\\fast.ai -- deep learning\\Lesson 1\\dogscats\\sample_small\\"

batches = vgg.get_batches(path + "train", batch_size = batch_size)
val_batches = vgg.get_batches(path + "valid", batch_size = batch_size*2)

imgs,labels = next(batches)
print("image shape " + str(imgs.shape))

# Note: The pretrained weights are from a network that has BGR as the input format
utils.plots(imgs, titles = labels)
plt.show()


################################################################################
## Finetune existing model
# https://keras.io/models/model/

vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch = 1)

################################################################################
## Make predictions
# VGG model returns 1,000 probabilities for each image

vgg.classes[:4]  # show first four class labels
pred = vgg.predict(imgs, True)
