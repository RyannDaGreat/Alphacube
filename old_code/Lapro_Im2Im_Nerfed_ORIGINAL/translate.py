import torch.nn.functional as F
import argparse
from utils import get_config
from data import ImageFolder, label2Color
from trainer import MUNIT_Trainer
import numpy as np
import torchvision.utils as vutils
import torchvision
import sys
import torch
import os
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/simulation2surgery.yaml', help='Path to the config file. Use this to adjust the output image size.')
parser.add_argument('--input_folder', type=str, nargs='+', help="input image folder(s)", required=True)
parser.add_argument('--output_folder', type=str, help="output image folder", required=True)
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders", required=True)
parser.add_argument('--style_input_folder', type=str, help="input image folder")
parser.add_argument('--seed', type=int, default=1, help="random seed (for drawing styles)")
parser.add_argument('--num_styles',type=int, default=5, help="number of styles to sample")
parser.add_argument('--store_style_images',action='store_true', help="store the style images which are used" )
#parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
#parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

opts = parser.parse_args()
conf = get_config(opts.config)
num_classes = conf["gen"]["num_classes"]

# Figure out how large the result images should be:
if "crop_image_width_translation" in conf:
    w = conf["crop_image_width_translation"]
else:
    w = conf["crop_image_width"]
if "crop_image_height_translation" in conf:
    h = conf["crop_image_height_translation"]
else:
    h = conf["crop_image_height"]
print("Will output translated images of size {}x{}".format(w,h))

if conf["batch_size"] > 1:
    print("Warning: Currently only batch size of 1 is supported during translation.")
    conf["batch_size"] = 1

if opts.store_style_images and not opts.style_input_folder:
    print("Warning: 'store_stlye_images' set, but no style_input_folder given. Will use random styles and not use style images.")

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

style_dim = conf['gen']['style_dim']
trainer = MUNIT_Trainer(conf)
state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()

aug = {}
aug["new_size_min"] = conf["new_size_min_a"]
aug["new_size_max"] = conf["new_size_max_a"]
aug["output_size"] = (w,h)
aug["circle_mask"] = False
aug["rotate"] = False
datasets = []
for path in opts.input_folder:
  translationData = ImageFolder(path, return_labels=False, return_paths=True, augmentation=aug)
  datasets.append( translationData )
concatDataset = torch.utils.data.ConcatDataset( datasets )
translationDataLoader = torch.utils.data.DataLoader(dataset=concatDataset, batch_size=conf["batch_size"], shuffle=False, drop_last=True, num_workers=conf["num_workers"])

fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

randStyle = True
if opts.style_input_folder:
    print("Using random style images from:", opts.style_input_folder)

    randStyle = False

    aug = {}
    aug["new_size_min"] = conf["new_size_min_b"]
    aug["new_size_max"] = conf["new_size_max_b"]
    aug["output_size"] = (w,h)
    aug["circle_mask"] = False
    aug["rotate"] = False
    styleData = ImageFolder(opts.style_input_folder, return_labels=False, return_paths=False, augmentation=aug)
    styleDataLoader = torch.utils.data.DataLoader(dataset=styleData, batch_size=conf["batch_size"], shuffle=True, drop_last=True, num_workers=conf["num_workers"])
    styleIterator = iter(styleDataLoader)

def getRandomStyleImage():
    global styleIterator
    try:
        sample = next(styleIterator)
    except StopIteration:
        styleIterator = iter(styleDataLoader)
        sample = next(styleIterator)
    return sample

def getStyleScore( seg_ab, lbl_a ):
    amax = torch.argmax( seg_ab, dim=1 )
    scorePerClass = []
    for classID in range( num_classes ):
        in_seg = torch.sum((amax == classID).int()).item()
        in_lbl = torch.sum((lbl_a == classID).int()).item()
        rel = abs((in_seg-in_lbl))
        scorePerClass.append( rel )
        #print( "\t", classID, in_seg, in_lbl, rel )
    mean = sum(scorePerClass)/len(scorePerClass)
    #print("Res style score:", mean )
    return mean


with torch.no_grad():
    imgCount = 0
    for sample in translationDataLoader:
        x_a = sample[0].cuda()

        c_a = trainer.gen_a.encode( x_a )

        path = sample[1][0]
        print("Input",path)
        seqname = os.path.basename( os.path.dirname( os.path.dirname( path ) ) )
        basename = os.path.basename(path)

        if randStyle:
            #results = []
            for style in range(0,opts.num_styles):
                s_b = torch.randn(conf["batch_size"], style_dim, 1, 1).cuda()
                x_ab = trainer.gen_b.decode( c_a, s_b )
                path = os.path.join(opts.output_folder, seqname, "style_%02d"%style, basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image((x_ab.data+1)/2, path, padding=0, normalize=False)
                #results.append( x_ab )
            #results = torch.cat( results, dim=0 )
            #filename="result{:04d}.png".format(imgCount)
            #vutils.save_image( results, os.path.join(opts.output_folder, filename), normalize=True )
        else:
            #for i in range(0,opts.num_style):
            for style in range(0,opts.num_styles):
                styleImg = getRandomStyleImage().cuda()

                c_b, s_b = trainer.gen_b.encode( styleImg )

                x_ab = trainer.gen_b.decode( c_a, s_b)

                path = os.path.join(opts.output_folder, seqname, "style_%02d"%style, basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))

                # Save the translated image:
                vutils.save_image((x_ab.data+1)/2, path, padding=0, normalize=False)

                # Optionally save the style image together with the translated image:
                if opts.store_style_images:
                    im = torch.cat( [styleImg,x_ab], dim=0 )
                    path = os.path.join(opts.output_folder, seqname, "style_%02d"%style, basename[:-4] + "_style_im.png" )
                    grid = vutils.make_grid( (im+1)/2, nrow=1, normalize=True, padding=0 )

                    pil_grid = torchvision.transforms.functional.to_pil_image( grid.cpu() )
                    vutils.save_image( torchvision.transforms.functional.to_tensor( pil_grid ), path )

                #print( imgCount, "Best score:", best_style_score, best_style_img_ind )

            #style_imgs = torch.cat( style_imgs, dim=0 )
            #results = torch.cat( results, dim=0 )
            #style_segmentations = torch.cat(style_segmentations)
            #style_segmentations = torch.argmax( style_segmentations, dim=1 ).unsqueeze(1).float()/num_classes

            #filename="input{:04d}.png".format(imgCount)
            #vutils.save_image( x_a, os.path.join(opts.output_folder, filename), normalize=True )
            #filename="style{:04d}.png".format(imgCount)
            #vutils.save_image( style_imgs, os.path.join(opts.output_folder, filename), normalize=True )
            #filename="result{:04d}.png".format(imgCount)
            #vutils.save_image( results, os.path.join(opts.output_folder, filename), normalize=True )
            #filename="best_result{:04d}.png".format(imgCount)
            #vutils.save_image( best_style_img, os.path.join(opts.output_folder, filename), normalize=True )
            #filename="segmentations{:04d}.png".format(imgCount)
            #vutils.save_image( style_segmentations, os.path.join(opts.output_folder, filename), normalize=True )

        # also save input images
        inp_dir = os.path.join( opts.output_folder, seqname, "input" )
        if not os.path.exists(inp_dir):
            os.makedirs(inp_dir)
        # copy the label file:
        #label_dir = os.path.join( opts.output_folder, seqname, "labels" )
        #if not os.path.exists(label_dir):
            #os.makedirs(label_dir)
        vutils.save_image(x_a.data, os.path.join(inp_dir, basename), padding=0, normalize=True)

        imgCount += 1
        #if imgCount > 20:
        #    break

