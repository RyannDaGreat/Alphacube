#TODO: Test this in a simple notebook, with a few textures rendered in blender
#The idea is to recreate the textures from a rendering+uv pair

import einops
import torch

def calculate_subpixel_weights(x,y):
    #This function's math is explained here: 
    #   https://www.desmos.com/calculator/esool5qrrd

    Rx=x%1
    Ry=y%1
    Qx=1-Rx
    Qy=1-Ry

    A=Rx*Ry #Weight for  ceil(x), ceil(y)
    B=Rx*Qy #Weight for  ceil(x),floor(y)
    C=Qx*Qy #Weight for floor(x),floor(x)
    D=Qx*Ry #Weight for floor(x), ceil(y)

    Cx=x.ceil ()
    Cy=y.ceil ()
    Fx=x.floor()
    Fy=y.floor()

    X=torch.stack((Cx,Cx,Fx,Fx)) #All X values
    Y=torch.stack((Cy,Fy,Fy,Cy)) #All Y values
    W=torch.stack((A ,B ,C ,D )) #Weights

    return X,Y,W
    
def unproject_translations(scene_uvs, scene_labels, scene_translations, num_labels, output_height, output_width):
    
    #------- Input Validation -------
    
    #Validate scene_uvs
    batch_size, two, scene_height, scene_width = scene_uvs.shape
    assert two == 2, "scene_uvs should have exactly two channels"
    assert scene_uvs.min() >= 0 and scene_uvs.max() <=1, 'All u,v values should be between 0 and 1'

    #Validate scene_labels
    assert scene_labels.shape == (batch_size, height, width)
    assert scene_labels.max() < num_labels, "The scene_labels includes labels that can't be included in the output!"

    #Validate scene_translations
    num_channels = scene_translations.shape[1]
    assert scene_translations.shape == (batch_size, num_channels, scene_height, scene_width)


    #------- Calculate Output -------



    u,v = einops.rearrange(scene_uvs,'B C H W -> C B H W')
    assert u.shape == v.shape == (batch_size, scene_height, scene_width)

    u,v,w = calculate_subpixel_weights(u,v)
    assert u.shape == v.shape == w.shape == (4, batch_size, scene_height, scene_width)

    weighted_scene_translations = scene_translations[None, :, :, :, :] * w

    u = einops.rearrange(


    



    #------- Output Validation -------

    #Validate scene_translations
    assert output_sum   .shape == (num_labels, num_channels, output_height, output_width)
    assert output_weight.shape == (num_labels,               output_height, output_width)

    
    
    
    
