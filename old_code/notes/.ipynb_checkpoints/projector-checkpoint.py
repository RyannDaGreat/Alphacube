#MAIN TODO:
#TODO: We need a way to visualize the blue channel. Perhaps we should have another function that assigns colors to each class and then turns that into an image? This could also be used as the second parameter in the network, and it would *learn* colors...
#...it's also a PERFECT way to verify that the blue values are indeed the values we think they are. We should test with maybe 5 or 6 objects...


# apply_albedo_via_numpy
# condense_values
# apply_albedos_via_numpy (see apply_albedo_via_numpy for a simpler version)
# apply_albedo_via_torch
# apply_albedos_via_torch (see apply_albedo_via_torch for a simpler version)

import numpy as np
import rp, torch

def apply_albedo_via_numpy(model,albedo):
    #See multitexture_test.ipynb for example usage
    
    assert rp.is_image(model)
    assert rp.is_rgb_image(albedo)
    assert rp.is_rgb_image(model) or rp.is_rgba_image(model)
    
    model=rp.as_float_image(model)
    
    albedo_height,albedo_width=rp.get_image_dimensions(albedo)
    model_height ,model_width =rp.get_image_dimensions(model )
    
    r=model[:,:,0]
    g=model[:,:,1]
    
    u=np.minimum((r*albedo_width ).astype(int),albedo_width -1)
    v=np.minimum((g*albedo_height).astype(int),albedo_height-1)
    
    output=albedo[v.flatten(),u.flatten()]
    output=output.reshape((model_height,model_width,3))
    
    #Black background stays black
    mask=1-(r==0)*(g==0)
    mask=np.expand_dims(mask,2)
    mask=mask.astype(np.uint8)
    output=mask*output
    
    return output

def condense_values(tensor,values):
    #Mutates the given tensor and returns it
    #   - tensor: can be either torch.Tensor or np.ndarray
    #   - values: should be a list of int
    #    
    # EXAMPLES:
    #     >>> condense_values(torch.Tensor([4,2,3,6,8,1]),[1,3,8])
    #    ans = tensor([4., 2., 1., 6., 2., 0.])
    #     >>> condense_values(torch.tensor([4,2,3,6,8,1]),[1,3,8])
    #    ans = tensor([4, 2, 1, 6, 2, 0])
    #     >>> condense_values(torch.tensor([4,2,3,6,8,1]),[1,2,3,4])
    #    ans = tensor([3, 1, 2, 6, 8, 0])
    #     >>> condense_values(torch.tensor([2,5,7,9]),[2,5,9])
    #    ans = tensor([0, 1, 7, 2])
    #     >>> condense_values(torch.tensor([2,5,7,9]),[2,5,7,9])
    #    ans = tensor([0, 1, 2, 3])
    #    
    #I'm having a hard time coming up with an intuitive explanation
    #for exactly what this function does...but know that it's used
    #to let us specify which values of the b channel we're using from
    #the model images, so we can use multiple textures efficiently
    #See the uses of this function for a better explanation...
    
    values=sorted(values)
    for index,value in enumerate(values):
        tensor[tensor==value]=index
    return tensor


def apply_albedos_via_numpy(model,albedos:dict):
    
    #NOTE: This has not been tested with albedo ID's other than 0 and 255 yet
    
    assert isinstance(albedos,dict)
    assert all(isinstance(key,int) for key in albedos)
    assert all(0<=key<=255 for key in albedos),'All albedo IDs should be encoded in the blue channel with values between 0 and 255'
    assert rp.is_image(model)
    assert all(rp.is_rgb_image(albedo) for albedo in albedos.values())
    assert rp.is_rgb_image(model) or rp.is_rgba_image(model)
    
    albedo_dimensions=set(rp.get_image_dimensions(albedo) for albedo in albedos.values())
    assert len(albedo_dimensions)==1,'All albedos should be the same size'
    albedo_dimensions=albedo_dimensions.pop()
    
    # model=as_byte_image(model) # Uncomment this line if you want to see how much worse the mapping is without EXR's
    model=rp.as_float_image(model)
    
    model_height,model_width=rp.get_image_dimensions(model)
    
    r=model[:,:,0]
    g=model[:,:,1]
    b=model[:,:,2]
    
    albedo_height,albedo_width=albedo_dimensions
    u=np.minimum((r*albedo_width ).astype(int),albedo_width -1)
    v=np.minimum((g*albedo_height).astype(int),albedo_height-1)
    i=np.minimum(np.round(b*256).astype(int),255) #Albedo ID
    i=condense_values(i,list(albedos))
    
    albedo=np.asarray([albedos[key] for key in sorted(albedos)])
    
    output=albedo[i.flatten(),v.flatten(),u.flatten()]
    output=output.reshape((model_height,model_width,3))
    
    #Black background stays black
    mask=1-(r==0)*(g==0)
    mask=np.expand_dims(mask,2)
    mask=mask.astype(np.uint8)
    output=mask*output
    
    return output


def apply_albedo_via_torch(models,albedo):
    
    assert len(albedo.shape)==3 and albedo.shape[2]==3,'Albedo shape should be like (512,512,3) or (256,128,3) etc'
    
    albedo_height,albedo_width=albedo.shape[:2]
    models_height,models_width=models.shape[2:]
    
    r=models[:,0,:,:]
    g=models[:,1,:,:]
    
    u=torch.clamp(r*albedo_height,min=0,max=albedo_height-1).type(torch.long)
    v=torch.clamp(g*albedo_width ,min=0,max=albedo_width -1).type(torch.long)
    
    output=albedo[v.flatten(),u.flatten()]
    output=output.reshape(r.shape+(-1,))
    output=output.permute(0,3,1,2)

    #Make black pixels in the model black in the rendering too
    mask=~((r==0)*(g==0))
    mask=mask.unsqueeze(1)
    output=output*mask
    
    return output


def apply_albedos_via_torch(models,albedos:dict):
    
    assert isinstance(albedos,dict)
    assert all(isinstance(key,int) for key in albedos)
    assert all(0<=key<=255 for key in albedos),'All albedo IDs should be encoded in the blue channel with values between 0 and 255'
    
    
    albedo_shapes=set(albedo.shape for albedo in albedos.values())
    assert len(albedo_shapes)==1,'All albedos should be the same size'
    albedo_shape=albedo_shapes.pop()
    assert len(albedo_shape)==3 and albedo_shape[2]==3,'Albedo shape should be like (512,512,3) or (256,128,3) etc'
    
    albedo_height,albedo_width=albedo_shape[:2]
    models_height,models_width=models.shape[2:]
    
    r=models[:,0,:,:]
    g=models[:,1,:,:]
    b=models[:,2,:,:]
    
    u=torch.clamp(r*albedo_height   ,min=0,max=albedo_height-1).type(torch.long)
    v=torch.clamp(g*albedo_width    ,min=0,max=albedo_width -1).type(torch.long)
    i=torch.clamp(torch.round(b*256),min=0,max=255            ).type(torch.long)
    i=condense_values(i,list(albedos))
    
    albedo=torch.stack([albedos[key] for key in sorted(albedos)])
    
    output=albedo[i.flatten(),v.flatten(),u.flatten()]
    output=output.reshape(r.shape+(-1,))
    output=output.permute(0,3,1,2)

    #Make black pixels in the model black in the rendering too
    mask=~((r==0)*(g==0))
    mask=mask.unsqueeze(1)
    output=output*mask
    
    return output
