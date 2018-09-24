"""
This script sets up the training data for the VOC 2010 "Person" dataset, as used in the paper 

  "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,
    and Fully Connected CRFs" (Chen et. al., 2017)

It assumes that you downloaded the annotations from:

   http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html

and unpacked them in a directory. You can then provide the path to the input directory in
the code below. The directory must be the one where the ".mat" files are located.

What this script does:
It goes over all .mat files, locates all objects, ignores all objects that are not "person",
extracts the masks for each part of the person, and merges them together. Partnames are
simplified from 24 classes to 6 classes as described by Chen et. al. It then saves the
summarized mask as a greyscale png image, where intesity 0 is background and all other
intesities (1-6) are the class labels.
"""

import scipy.io as sio
import os
import imageio
import numpy as np


part_simplification =  {
 'hair': 'head',
 'head': 'head',
 'lear': 'head',
 'lebrow': 'head',
 'leye': 'head',
 'mouth': 'head',
 'nose': 'head',
 'rear': 'head',
 'rebrow': 'head',
 'reye': 'head',
 'lhand': 'lowerarm',
 'llarm': 'lowerarm',
 'rhand': 'lowerarm',
 'rlarm': 'lowerarm',
 'llleg': 'lowerleg',
 'rfoot': 'lowerleg',
 'rlleg': 'lowerleg',
 'lfoot': 'lowerleg',
 'luarm': 'upperarm',
 'ruarm': 'upperarm',
 'luleg': 'upperleg',
 'ruleg': 'upperleg',
 'neck': 'torso',
 'torso': 'torso'
}

part_ids = {'head':1, 'torso':2, 'upperarm':3, 'lowerarm':4, 'upperleg':5, 'lowerleg':6 }

data_dir = "/home/dermaniac/data/VOCdevkit/VOC2010/person_trainval/Annotations_Part/"
output_dir = "/home/dermaniac/data/VOCdevkit/VOC2010/person_trainval/Annotations_Part_images/"

for file in os.listdir(data_dir):
    gt = sio.loadmat(data_dir+"/"+file)["anno"]
    objects = gt[0][0][1][0]
    
    simple_gt = None
    print(file)
    for i in range(objects.shape[0]):
        if objects[i][0][0]=="person":
            if(objects[i][3].shape[0] > 0):
                parts = objects[i][3][0]
                for p in range(parts.shape[0]):
                    part = parts[p]
                    partname = part[0][0]
                    general_part = part_simplification[partname]
                    part_id = part_ids[general_part]
                    if simple_gt is None:
                        simple_gt = np.zeros(part[1].shape, dtype="uint8")
                    simple_gt[part[1]==1] = part_id
    
    if not simple_gt is None:
        # Filter out files where there is a person object in the mat file, but 
        # but nowhere in the image
        if np.sum(simple_gt) > 0:
            outfilename = file[:-4]+".png"
            imageio.imwrite(output_dir+"/"+outfilename, simple_gt)


