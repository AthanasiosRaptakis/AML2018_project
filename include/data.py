import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import skimage.io as io
from PIL import Image
from scipy.misc import imresize
from inferno.io.transform.generic import Label2OneHot

class ObjectSegmentationDataset(Dataset):
    def __init__(self, src_image_dir, seg_image_dir, num_classes, transform=None, rescale=True, augment=True, gt_one_hot=False):
        self.transform = transform
        self.num_classes = num_classes
        self.src_image_dir = src_image_dir
        self.seg_image_dir = seg_image_dir
        self.gt_one_hot = gt_one_hot
        if(rescale==True):
            rescale = (250,250)
        self.rescale = rescale
        self.augment = augment
        # store names without file extension
        self.image_names = [name[:-4] for name in os.listdir(self.seg_image_dir)]
            
    def __len__(self):
        if self.augment:
            if self.rescale != False:
                # If we rescale to 4, then random rescaling for augmentation makes no sense
                return len(self.image_names) * 4
            else:
                return len(self.image_names) * 8
        else:
            return len(self.image_names)
    
    def __getitem__(self, idx):
        horizontal_mirror = False
        additive_noise = False
        random_rescale = False
        
        if self.augment:
            if (idx % 2) == 1:
                horizontal_mirror = True
            idx = idx // 2
            if (idx % 2) == 1:
                additive_noise = True
            idx = idx // 2
            # If we don't rescale to 250x250, we can perform dataset augmentation
            # using random rescaling
            if self.rescale==False:
                if (idx % 2) == 1:
                    random_rescale = True
                idx = idx // 2

        src_file_path = os.path.join(self.src_image_dir, self.image_names[idx])+".jpg"
        seg_file_path = os.path.join(self.seg_image_dir, self.image_names[idx])+".png"
        
        src_image = io.imread(src_file_path).astype("uint8")

        if additive_noise:
            noise = np.random.uniform(0,1,(src_image.shape[0],src_image.shape[1])) > 0.01
            src_image[:,:,0] = src_image[:,:,0] * noise
            src_image[:,:,1] = src_image[:,:,1] * noise
            src_image[:,:,2] = src_image[:,:,2] * noise
        
        # Read seg_image using PIL in order to preserve the color index from the pallette
        # These are actually our class labels
        seg_image = Image.open(seg_file_path)
        
        
        if self.rescale != False:
            src_image = imresize(src_image,self.rescale,interp='bilinear')
            seg_image = transforms.functional.resize(seg_image,self.rescale,0)
            
        if random_rescale:
            hnew = int(src_image.shape[0] * np.random.uniform(low=0.5, high=1.5))
            wnew = int(src_image.shape[1] * np.random.uniform(low=0.5, high=1.5))
            src_image = imresize(src_image,(hnew,wnew),interp='bilinear')
            seg_image = transforms.functional.resize(seg_image,(hnew,wnew),0)

        seg_image_ds = transforms.functional.resize(seg_image, (src_image.shape[0]//8,src_image.shape[1]//8),0)

        seg_image_ds = np.array(seg_image_ds)
        seg_image = np.array(seg_image)
        

        # Change 255 to num_classes-1 so there are not gaps in the label numbers
        seg_image_ds[seg_image_ds == 255] = self.num_classes - 1
        seg_image[seg_image == 255] = self.num_classes - 1

        if horizontal_mirror:
            src_image = np.flip(src_image, axis=1).copy()
            seg_image = np.flip(seg_image, axis=1).copy()
            seg_image_ds = np.flip(seg_image_ds, axis=1).copy()
        
        if(self.gt_one_hot):
            seg_image = Label2OneHot(self.num_classes)(seg_image)
            seg_image_ds = Label2OneHot(self.num_classes)(seg_image_ds)
        seg_image = torch.Tensor(seg_image).int()
        seg_image_ds = torch.Tensor(seg_image_ds).int()

        if self.transform:
            src_image_trans = self.transform(src_image.copy()) # Need to copy because some transformations are in-place
        else:
            src_image_trans = src_image.copy()
            

        # The CRF requires the not-normalized image, so I implemented the option
        # to also get both the normalized and not-normalized image
        return (src_image_trans, seg_image, seg_image_ds, src_image)

