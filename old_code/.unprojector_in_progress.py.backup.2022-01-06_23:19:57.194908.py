#TODO: Test this in a simple notebook, with a few textures rendered in blender
#   An animation of a cube rotating, along with the UV texture maps revealed would be pretty cool to see
#TODO: Make this work with requires_grad=True
#TODO: Make this work with torch.Tensor.index_add_(...)
#The idea is to recreate the textures from a rendering+uv pair

import einops
import torch
import icecream
import time

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

def unproject_translations(scene_uvs, scene_labels, scene_translations, num_labels:int, output_height:int, output_width:int,version='slow'):
    # This is a pure function: it doesn't mutate any of the input tensors
    # MAYBE todo: eliminate batch size, because we're going to be doing this once for every image anyway...
    # Batch size is technically unnesecary...but it might be nice to keep it, idk...it combines confidence of multiple scenes very nicely...(ok, I'll keep batch size)

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

    #This 'Calculate Output' section is acronym-heavy.
    #Acronym Key:
    #  - BS = batch_size
    #  - SH,SW,OH,OW = scene_height,scene_width,output_height,output_width
    #  - NC,NL = num_channels,num_labels
    #  - ABCD = 4: it refers to the four A,B,C,D values in the function calculate_subpixel_weights
    #  - w = weights
    #  - t = weighted scene_translations, which are summed together to get the output
    BS = batch_size
    SH = scene_height
    SW = scene_width
    NC = num_channels
    NL = num_labels
    OH = output_height
    OW = output_width

    #Populate the outputs
    u,v = einops.rearrange(scene_uvs.clone(),'BS NC SH SW -> NC BS SH SW')
    assert u.shape == v.shape == (batch_size, scene_height, scene_width)
    
    #Convert u,v from range [0,1),[0,1) to [0,output_height),[0,output_width)
    u *= output_height#-1
    v *= output_width #-1 To subtract 1 or not...depends on the projection function, and whether not u=.999999 means a bottom pixel

    #Calculate the subpixel weights w, and make u,v integers
    u,v,w = calculate_subpixel_weights(u,v)
    assert u.shape == v.shape == w.shape == (4, batch_size, scene_height, scene_width)
    assert not u.dtype.is_floating_point and not v.dtype.is_floating_point

    #The calculate_subpixel_weights is likely to reference indices one index outside of the image.
    #Correct this by pushing those indices back into the image again.
    u=u.clamp(0,output_height-1)
    v=v.clamp(0,output_width -1)
    l = scene_labels

    t = scene_translations

    #Reshape stuff
    u = einops.rearrange(u, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    v = einops.rearrange(v, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    w = einops.rearrange(w, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    l = einops.repeat   (l, '     BS    SH SW -> (BS SH SW ABCD)   ',ABCD=4)
    t = einops.repeat   (t, '     BS NC SH SW -> (BS SH SW ABCD) NC',ABCD=4)

    #Create empty output tensors
    output_device = scene_translations.device
    output_weight = torch.zeros((num_labels, output_height, output_width              ), device = output_device)
    output_sum    = torch.zeros((num_labels, output_height, output_width, num_channels), device = output_device)

    assert len(t) == len(u) == len(v) == len(w) == len(l)

    #TODO: Perhaps keep track of output_sum_of_squares as well (another vector) so that we could output the variance at the same time as we output the sum?
    #The alternative is running this funtion multiple times, and comparing the texture outputs...which is also a totally valid approach. It doesn't need BS to do that though...

    if version=='slow':
        #THE SUPER-SLOW VERSION:    
        # This version is very slow, but easier to understand. 
        # It gives exactly the same results as when version=='fast'
        for l_,u_,v_,t_,w_ in zip(l,u,v,t,w):
            output_sum   [l_,u_,v_]+=w_*t_
            output_weight[l_,u_,v_]+=w_
        
    
    if version=='fast':
        #THE FAST VERSION:
        #This version is very fast, but a bit cryptic
        #
        #Summary of this version using imprecise notation:        
        #  #What I'd like to write, but can't:
        #     output_sum   [l, u, v] += t 
        #     output_weight[l, u, v] += w 
        #     # This doesn't accumulate properly when we have duplicate l,u,v's
        #  #Splitting it into R,G,B:
        #     output_sum   [l, u, v, 0] += t[:,0] 
        #     output_sum   [l, u, v, 1] += t[:,1] 
        #     output_sum   [l, u, v, 2] += t[:,2] 
        #     output_weight[l, u, v] += w 

        output_sum=output_sum.view(NL*OH*OW*NC)
        for i in range(NC):
            output_sum.index_add_(0, l*OH*OW*NC + u*OW*NC + v*NC + i, t[:,i]*w)
        output_sum=output_sum.view(NL,OH,OW,NC)
        
        output_weight=output_weight.view(NL*OH*OW)
        output_weight.index_add_(0, l*OH*OW + u*OW + v, w)
        output_weight=output_weight.view(NL,OH,OW)

    output_sum = einops.rearrange(output_sum, 'NL OH OW NC -> NL NC OH OW')
    assert output_sum.shape == (num_labels, num_channels, output_height, output_width)
    
    denominator = output_weight.clone()
    denominator[denominator == 0] = 1  # Avoid division by zero errors
    denominator = denominator[:, None, :, :]  # Make this tensor fit

    output_mean = output_sum/denominator



    #------- Output Validation -------

    assert output_weight.shape == (num_labels, output_height, output_width)

    
    assert output_mean.shape == output_sum.shape
    
    return output_mean, output_weight
    
def test_unproject_translations():
    C = 3
    L = 5
    B = 3
    SH = 256
    SW = 256
    OH = 256
    OW = 256
    
    ra = torch.rand
    ri = lambda *size: torch.randint(0, L - 4, size)
    
    t=ra(B, C, SH, SW).cuda()
    t.requires_grad=True
    
    su=ra(B, 2, SH, SW).cuda()
    sl=ri(B,    SH, SW).cuda()
    
    #Make sure the slow and fast versions return the same values
    start_time=time.time()
    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='fast'
    )
    print('FAST VERSION OUTPUT: took %f seconds'%(time.time()-start_time))
    print('(outputs should be same as SLOW):')
    icecream.ic(out_tex[0,0],out_weight[0,0])
    
    
    start_time=time.time()
    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='slow'
    )
    
    print('SLOW VERSION OUTPUT: took %f seconds'%(time.time()-start_time))
    icecream.ic(out_tex[0,0],out_weight[0,0])

    print('RESULTS:')
    icecream.ic(C,L,B,SH,SW,OH,OW,out_tex.shape,out_weight.shape)
    icecream.ic(t.grad)
    out_tex.sum().backward()
    icecream.ic(t.grad.shape)

test_unproject_translations()
