"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
import os
import os.path
import random
from collections import namedtuple

import numpy
import torch
import torch.utils.data as data
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if not root.endswith( "ids" ) and not root.endswith( "labels" ) and not root.endswith( "normals" ) and not root.endswith( "depths" ):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def circleMask( img, ox, oy, radius ):
    mask = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(mask)
    x0 = img.size[0]*0.5 - radius + ox
    x1 = img.size[0]*0.5 + radius + ox
    y0 = img.size[1]*0.5 - radius + oy
    y1 = img.size[1]*0.5 + radius + oy
    draw.ellipse([x0,y0,x1,y1], fill=0)
    img.paste( (0,0,0), mask=mask )
    return img


class ImageFolder(data.Dataset):

    def __init__(
        self,
        root,
        loader        = default_loader,
        return_paths  = False,
        augmentation  = {},
    ):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root            = root
        self.imgs            = imgs
        self.return_paths    = return_paths
        self.loader          = loader
        self.output_size     = augmentation["output_size"]
        self.add_circle_mask = "circle_mask" in augmentation and augmentation["circle_mask"] == True
        self.rotate          = "rotate"      in augmentation and augmentation["rotate"]      == True
        self.contrast        = "contrast"    in augmentation and augmentation["contrast"]    == True

        if "new_size_min" in augmentation and "new_size_max" in augmentation:
            self.new_size_min = augmentation["new_size_min"]
            self.new_size_max = augmentation["new_size_max"]
        else:
            self.new_size_min = min(self.output_size)
            self.new_size_max = min(self.output_size)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)

        maskRadius, maskOx, maskOy = None, None, None

        randAng = random.random()*20-10
        minOutputSize = min(self.output_size)
        maxOutputSize = max(self.output_size)
        randSize = random.randint( self.new_size_min, self.new_size_max )

        if self.add_circle_mask:
            minSize = min((img.width, img.height))
            maxSize = max((img.width, img.height))

            maxRadius = int(math.sqrt((img.width/2)**2 + (img.height/2)**2))
            minRadius = int(0.4*maxSize)
            
            maskRadius = random.randint( minRadius, maxRadius )

            maskOx = random.randint(int(-img.width  * 0.1), int(img.width  * 0.1))
            maskOy = random.randint(int(-img.height * 0.1), int(img.height * 0.1))

            img = circleMask(img, maskOx, maskOy, maskRadius)

        img = transforms.functional.resize( img, randSize, Image.BILINEAR )
        if self.rotate:
            img = transforms.functional.rotate( img, randAng, Image.BILINEAR )
        
        rx = random.randint( 0, max(img.width  - self.output_size[0],0) )
        ry = random.randint( 0, max(img.height - self.output_size[1],0) )

        img = transforms.functional.crop( img, ry, rx, self.output_size[1], self.output_size[0] )

        img = transforms.functional.to_tensor( img )
        if self.contrast:
            c = random.uniform( 0.75, 1.25 )
            b = random.uniform( -0.1, 0.1 )
            img = img*c + b

        img = transforms.functional.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
