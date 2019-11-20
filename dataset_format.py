from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import datasets
import shutil
import random
from os import path

name_set = "caltech101"

root_data_loc = "data/" + name_set
data_dir = "data/" + name_set + "_all"
train = "data/" + name_set + "/train"
val = "data/" + name_set + "/val"
train_frac = 0.8

if __name__ == '__main__':

    # delete / create folders
    print("deleting and then creating folders")
    if os.path.exists(root_data_loc): # reset root
        shutil.rmtree(root_data_loc)
    if os.path.exists(train): # rmv old
        shutil.rmtree(train)
    if os.path.exists(val):
        shutil.rmtree(val)
    os.mkdir(root_data_loc) # make everything again
    os.mkdir(train)
    os.mkdir(val)

    # iter over categories/labels
    for label in os.listdir(data_dir):
        print(f"processing label: {label}")
        label_path = os.path.join(data_dir, label)
        os.mkdir(path.join(train,label))
        os.mkdir(path.join(val, label))
        if os.path.isdir(label_path): # check is directory
            files = [file for file in os.listdir(label_path)]
            random.shuffle(files)
            numTrain = round(len(files) * train_frac)
            for f in files[:numTrain]:
                dst =  path.join(path.join(train, label), f)
                shutil.copyfile(path.join(label_path, f), dst)
            for f in files[numTrain:]:
                dst = path.join(path.join(val, label), f)
                shutil.copyfile(path.join(label_path, f), dst)
