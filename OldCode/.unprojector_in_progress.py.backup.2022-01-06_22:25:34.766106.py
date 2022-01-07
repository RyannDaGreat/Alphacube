#TODO: Test this in a simple notebook, with a few textures rendered in blender
#   An animation of a cube rotating, along with the UV texture maps revealed would be pretty cool to see
#TODO: Make this work with requires_grad=True
#TODO: Make this work with torch.Tensor.index_add_(...)
#The idea is to recreate the textures from a rendering+uv pair

import einops
import torch
import icecream

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

    #TODO: Can weighing of t be brought to the line that looks like "output_sum   [l[i], u[i], v[i]] += t[i]" and instead we have a new line that's just "t=scene_translations"?
    #                         1,bs,nc,sh,sw  *  4,bs,1,sh,sw     TODO: Replace the next line with einsum
    t = scene_translations[None, :, :, :, :] * w[:, :, None, :, :] # weighted scene_translations
    assert t.shape == (4, batch_size, num_channels, scene_height, scene_width)
    
    l = scene_labels

    #Reshape stuff
    t = einops.rearrange(t, 'ABCD BS NC SH SW -> (BS SH SW ABCD) NC'       )
    u = einops.rearrange(u, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    v = einops.rearrange(v, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    w = einops.rearrange(w, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    l = einops.repeat   (l, '     BS    SH SW -> (BS SH SW ABCD)   ',ABCD=4)

    #Create empty output tensors
    output_device = scene_translations.device
    output_weight = torch.zeros((num_labels, output_height, output_width              ), device = output_device)
    output_sum    = torch.zeros((num_labels, output_height, output_width, num_channels), device = output_device)

    assert len(t) == len(u) == len(v) == len(w) == len(l)

    #TODO: Perhaps keep track of output_sum_of_squares as well (another vector) so that we could output the variance at the same time as we output the sum?
    #The alternative is running this funtion multiple times, and comparing the texture outputs...which is also a totally valid approach. It doesn't need BS to do that though...

    #THE SUPER-SLOW METHOD:    
    if version=='slow':
        for l_,u_,v_,t_,w_ in zip(l,u,v,t,w):
            output_sum   [l_,u_,v_]+=t_
            output_weight[l_,u_,v_]+=w_
        
    #THE BROKEN METHOD:
    #output_sum   [l, u, v] += t 
    #output_weight[l, u, v] += w 
    
    if version=='fast':
        # THE RGB VERSION OF THE BROKEN METHOD:
        NL = num_labels
        OH = output_height
        OW = output_width
        NC = num_channels

        #Summary of this method using imprecise notation:        
        #     output_sum   [l, u, v, 0] += t[:,0] 
        #     output_sum   [l, u, v, 1] += t[:,1] 
        #     output_sum   [l, u, v, 2] += t[:,2] 
        #     output_weight[l, u, v] += w 

        output_sum=output_sum.view(NL*OH*OW*NC)
        output_sum.index_add_(0,l*OH*OW*NC + u*OW*NC + v*NC + 0,t[:,0])
        output_sum.index_add_(0,l*OH*OW*NC + u*OW*NC + v*NC + 1,t[:,1])
        output_sum.index_add_(0,l*OH*OW*NC + u*OW*NC + v*NC + 2,t[:,2])
        output_sum=output_sum.view(NL,OH,OW,NC)
        
        output_weight=output_weight.view(NL*OH*OW)
        output_weight.index_add_(0,l*OH*OW + u*OW + v,w)
        output_weight=output_weight.view(NL,OH,OW)





    #output_sum.requires_grad=True


    output_sum = einops.rearrange(output_sum, 'NL OH OW NC -> NL NC OH OW')
    
    #for i in range(batch_size):
    #    assert len(u[i]) == len(v[i]) == len(t[i])#NOTE: This for-loop should be eliminatable if we concatenate all the indices of l[i], t[i] along the batch dimension! This would let us use a single index_add operation.
    #    #....but why is that ok? It's ok because, remember, we're combining all the samples from the batch together. The output doesn't have BS in it's dimensions: it only has NL. 
    #    output_sum   [l[i], u[i], v[i]] += t[i] #The only difference between this and straightup assignment is that we might have repeating indices
    #    output_weight[l[i], u[i], v[i]] += w[i] #So, if we can generate each of these sums then stack them together we can do this all in-place.
    #    #TODO: do this all with index_sum_ (or some helper function)
    #    #############NOTE: Tensor.index_add as opposed to Tensor.index_add_ is NOT in-place, and takes exactly the same arguments! It's a substitute!
    #    #Until this is done, it will not perform as well as it could
    #output_sum = einops.rearrange(output_sum, 'NL OH OW NC -> NL NC OH OW')
    
    #TODO: Somehow modify torch.Tensor.index_sum_ to work like https://git.io/JS9MU should work (but doesn't)
    

    #------- Output Validation -------
    #
    assert output_sum   .shape == (num_labels, num_channels, output_height, output_width)
    assert output_weight.shape == (num_labels,               output_height, output_width)

    denominator = output_weight.clone()
    denominator[denominator==0] = 1 #Avoid division by zero errors
    denominator=denominator[:,None,:,:] #Make this tensor fit 
    output_mean = output_sum/denominator
    
    assert output_mean.shape == output_sum.shape
    
    return output_mean, output_weight
    
def test_unproject_translations():
    C = 3
    L = 5
    B = 3
    SH = 9
    SW = 7
    OH = 2
    OW = 3
    
    ra = torch.rand
    ri = lambda *size: torch.randint(0, L - 4, size)
    
    t=ra(B, C, SH, SW).cuda()
    t.requires_grad=True
    
    su=ra(B, 2, SH, SW).cuda()
    sl=ri(B,    SH, SW).cuda()
    
    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='slow'
    )
    print('SLOW-'*30)
    #print(out_tex,out_weight)
    print(out_tex[0,0],out_weight[0,0],'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])

    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='slow'
    )
    print('SLOW-'*30)
    #print(out_tex,out_weight)
    #print(out_tex,out_weight,su[...,0],sl[...,0])
    print(out_tex[0,0],out_weight[0,0],'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])

    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='fast'
    )
    print('FAST-'*30)
    #print(out_tex,out_weight,'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])
    print(out_tex[0,0],out_weight[0,0],'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])
    
    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='fast'
    )
    print('FAST-'*30)
    print(out_tex[0,0],out_weight[0,0],'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])
    #print(out_tex,out_weight,t)

    out_tex,out_weight=unproject_translations(
        su,
        sl,
        t,
        L,
        OH,
        OW,
        version='slow'
    )
    print('SLOW-'*30)
    #print(out_tex,out_weight)
    #print(out_tex,out_weight,su[...,0],sl[...,0])
    print(out_tex[0,0],out_weight[0,0],'\n------',su[0,0,0],sl[0,0,0],t[0,0,0])

    
    icecream.ic(C,L,B,SH,SW,OH,OW,out_tex.shape,out_weight.shape)
    icecream.ic(t.grad)
    out_tex.sum().backward()
    icecream.ic(t.grad.shape)
    

test_unproject_translations()
