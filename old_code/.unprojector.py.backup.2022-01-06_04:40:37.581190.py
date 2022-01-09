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

    Cx=x.ceil ().long()
    Cy=y.ceil ().long()
    Fx=x.floor().long()
    Fy=y.floor().long()

    X=torch.stack((Cx,Cx,Fx,Fx)) #All X values
    Y=torch.stack((Cy,Fy,Fy,Cy)) #All Y values
    W=torch.stack((A ,B ,C ,D )) #Weights

    return X,Y,W
    
def unproject_translations(scene_uvs, scene_labels, scene_translations, num_labels:int, output_height:int, output_width:int):
    
    #------- Input Validation -------
    
    #Validate scene_uvs
    batch_size, two, scene_height, scene_width = scene_uvs.shape
    assert two == 2, "scene_uvs should have exactly two channels"
    assert scene_uvs.min() >= 0 and scene_uvs.max() <=1, 'All u,v values should be between 0 and 1'

    #Validate scene_labels
    assert scene_labels.shape == (batch_size, scene_height, scene_width)
    assert scene_labels.max() < num_labels, "The scene_labels includes labels that can't be included in the output!"
    assert not scene_labels.dtype.is_floating_point, "scene_labels should be an integer type, because it is used as an index"

    #Validate scene_translations
    num_channels = scene_translations.shape[1]
    assert scene_translations.shape == (batch_size, num_channels, scene_height, scene_width)


    #------- Calculate Output -------

    #Create empty output tensors
    output_weight = torch.zeros((num_labels,               output_height, output_width))
    output_sum    = torch.zeros((num_labels, num_channels, output_height, output_width), 
                                requires_grad = True                                   ,
                                device        = scene_translations.device              )
    
    #The rest of this 'Calculate Output' section is acronym-heavy.
    #Acronym Key:
    #  - BS = batch_size
    #  - SH,SW,OH,OW = scene_height,scene_width,output_height,output_width
    #  - NC,NL = num_channels,num_labels
    #  - ABCD = 4: it refers to the A,B,C,D values in the function calculate_subpixel_weights
    #  - w = weights
    #  - t = weighted scene_translations, which are summed together to get the output

    #Populate the outputs
    u,v = einops.rearrange(scene_uvs,'BS NC SH SW -> NC BS SH SW')
    assert u.shape == v.shape == (batch_size, scene_height, scene_width)
    
    #Convert u,v from range [0,1),[0,1) to [0,output_height),[0,output_width)
    u *= output_height#-1
    v *= output_width #-1 To subtract 1 or not...depends on the projection function, and whether not u=.999999 means a bottom pixel

    u,v,w = calculate_subpixel_weights(u,v)
    assert u.shape == v.shape == w.shape == (4, batch_size, scene_height, scene_width)
    assert not u.dtype.is_floating_point and not v.dtype.is_floating_point

    #                         1,bs,nc,sh,sw  *  4,bs,1,sh,sw     TODO: Replace the next line with einsum
    t = scene_translations[None, :, :, :, :] * w[:, :, None, :, :] # weighted scene_translations
    assert t.shape == (4, batch_size, num_channels, scene_height, scene_width)
    
    l = scene_labels

    t = einops.rearrange(t, 'ABCD BS NC SH SW -> BS (SH SW ABCD) NC'       )
    u = einops.rearrange(u, 'ABCD BS    SH SW -> BS (SH SW ABCD)   '       )
    v = einops.rearrange(v, 'ABCD BS    SH SW -> BS (SH SW ABCD)   '       )
    w = einops.rearrange(w, 'ABCD BS    SH SW -> BS (SH SW ABCD)   '       )
    l = einops.repeat   (l, '     BS    SH SW -> BS (SH SW ABCD)   ',ABCD=4)

    output_sum = einops.rearrange(output_sum, 'NL NC OH OW -> NL OH OW NC')

    for i in range(batch_size):
        assert len(u[i]) == len(v[i]) == len(t[i])
        output_sum   [l[i], u[i], v[i]] += t[i]
        output_weight[l[i], u[i], v[i]] += w[i]
        #TODO: do this all with index_sum_ (or some helper function)
    
    #TODO: Somehow modify torch.Tensor.index_sum_ to work like https://git.io/JS9MU should work (but doesn't)
    #I'll implement it the wrong way below...
    
    #------- Output Validation -------

    assert output_sum   .shape == (num_labels, num_channels, output_height, output_width)
    assert output_weight.shape == (num_labels,               output_height, output_width)

    denominator = output_weight.clone()
    denominator[denominator==0] = 1 #Avoid division by zero errors
    denominator=denominator[:,None,:,:] #Make this tensor fit 
    output_mean = output_sum/denominator
    
    assert output_mean.shape == output_sum.shape
    
    return output_mean, output_weight
    
    

    