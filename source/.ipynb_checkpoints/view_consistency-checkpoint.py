import torch.nn as nn
import torch
from .unprojector import unproject_translations_individually


def weighted_mean(images, weights):
    #Takes BCHW images and BHW weights and returns a pixel-wise CHW average
    
    assert len(images.shape)==4 and len(weights.shape)==3
    
    numerator   = (images*weights[:,None]).sum(dim=0) # CHW tensor
    denominator = weights.sum(dim=0)[None]            # 1HW tensor
    
    denominator[denominator==0] = 1 #Get rid of 0's in denominator
    # When a particular pixel has 0 weight in any image, the output pixel is black
    
    return numerator/denominator


def weighted_variance(images, weights):
    #Takes BCHW images and BHW weights and returns a pixel-wise CHW variances
    #Variance is the average squared distance to the mean
    
    mean = weighted_mean(images, weights) #  CHW tensor
    mean = mean[None]                     # 1CHW tensor
    diff = images - mean                  # BCHW tensor
    squared_diff = diff ** 2              # BCHW tensor
    
    numerator   = (squared_diff*weights[:,None]).sum(dim=0) # CHW tensor
    denominator = weights.sum(dim=0)[None]                  # 1HW tensor
    
    denominator[denominator==0] = 1 #Get rid of 0's in denominator
    
    return numerator/denominator


class ViewConsistencyLoss(nn.Module):
    def __init__(self, recovery_width:int=256, recovery_height:int=256, version='std'):
        
        #Usually recovery_width==recovery_height. There's not really a good reason not to make it square
        
        super().__init__()
        
        assert version in ['std','var']
        
        self.version        =version
        self.recovery_width =recovery_width 
        self.recovery_height=recovery_height
        
    def forward(self, scene_translations, scene_uvs, scene_labels):
        
        #Calculate num_labels; it's one less argument we need to specify
        #We're just calculating a loss; it's ok to lose a texture we never see
        num_labels=1+scene_labels.max()
        
        recovered_texture_packs, recovered_weight_packs = unproject_translations_individually(scene_translations  ,
                                                                                              scene_uvs           ,
                                                                                              scene_labels        ,
                                                                                              num_labels          ,
                                                                                              self.recovery_height,
                                                                                              self.recovery_width )
        
        variances=[]
        for i in range(num_labels):
            variances.append(weighted_variance(recovered_texture_packs[:,i],recovered_weight_packs[:,i]))
        variances=torch.stack(variances)

        #The actual value of the output doesn't have much meaning. However, its gradient definitely does.
        if self.version=='var': return torch.mean(variances    )        
        if self.version=='std': return torch.mean(variances**.5)
    
        assert False, 'This line is unreachable' 