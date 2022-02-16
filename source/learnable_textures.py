import torch
import torch.nn as nn
import numpy as np
import rp


#This file contains three types of learnable images:
#    Raster: A simple RGB pixel grid
#    MLP: A per-pixel MLP that takes in XY and outputs RGB
#    Fourier: An MLP with fourier-feature inputs


##################################
######## HELPER FUNCTIONS ########
##################################


class GaussianFourierFeatureTransform:
    """
    Original authors: https://github.com/ndahlquist/pytorch-fourier-feature-networks
    
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def __call__(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    
    
def get_xy_grid(height:int, width:int, batch_size:int=1)->torch.Tensor:
    #Returns a torch cpu tensor of shape (batch_size,2,height,width)
    #Note: batch_size can probably be removed from this function after refactoring this file. It's always 1 in all usages.
    #The second dimension is (x,y) coordinates, which go from [0 to 1) from edge to edge
    #(In other words, it will include x=y=0, but instead of x=y=1 the other corner will be x=y=.999)
    #(this is so it doesn't wrap around the texture 360 degrees)
    assert height>0 and width>0 and batch_size>0,'All dimensions must be positive integers'
    
    y_coords = np.linspace(0, 1, height, endpoint=False)
    x_coords = np.linspace(0, 1, width , endpoint=False)
    
    xy_grid = np.stack(np.meshgrid(y_coords, x_coords), -1)
    xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous()
    xy_grid = xy_grid.repeat(batch_size,1,1,1)
    
    assert tuple(xy_grid.shape)==(batch_size,2,height,width)
    
    return xy_grid


##################################
######## LEARNABLE IMAGES ########
##################################


class LearnableImage(nn.Module):
    def __init__(self,
                 height      :int,
                 width       :int,
                 num_channels:int):

        #This is an abstract class, and is meant to be subclassed before use
        #Upon calling forward(), retuns a tensor of shape (num_channels, height, width)

        super().__init__()
        
        self.height      =height      
        self.width       =width       
        self.num_channels=num_channels
    
    def as_numpy_image(self):
        image=self()
        image=rp.as_numpy_array(image)
        image=image.transpose(1,2,0)
        return image

    
class LearnableImageRaster(LearnableImage):
    def __init__(self,
                 height      :int  ,
                 width       :int  ,
                 num_channels:int=3):
        
        super().__init__(height,width,num_channels)
        
        #An image paramterized by pixels

        self.image=nn.Parameter(torch.rand(num_channels,height,width))
        
    def forward(self):
        output = self.image
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    
class LearnableImageMLP(LearnableImage):
    def __init__(self,
                 height      :int     , # Height of the learnable images
                 width       :int     , # Width of the learnable images
                 num_channels:int=3   , # Number of channels in the images
                 hidden_dim  :int=256 , # Number of dimensions per hidden layer of the MLP
                 device      :int=None, # Please set this correctly if you want GPU
                ):
        
        super().__init__(height,width,num_channels)
        
        self.hidden_dim  =hidden_dim
        self.device      =device or 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # The following Tensor is NOT a parameter, and is not changed while optimizing this class
        self.xy_grid=get_xy_grid(height,width,batch_size=1).to(self.device)
        
        H=hidden_dim    # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=num_channels  # Shorter variable names let us align the code better
        self.model = nn.Sequential(
                nn.Conv2d(2, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, C, kernel_size=1),
                nn.Sigmoid(),
            ).to(self.device)
            
    def forward(self):
        output = self.model(self.xy_grid).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    
class LearnableImageFourier(LearnableImage):
    def __init__(self,
                 height      :int=256 , # Height of the learnable images
                 width       :int=256 , # Width of the learnable images
                 num_channels:int=3   , # Number of channels in the images
                 hidden_dim  :int=256 , # Number of dimensions per hidden layer of the MLP
                 mapping_size:int=128 , # Number of fourier features per coordinate
                 scale       :int=10  , # Magnitude of the initial feature noise
                 device      :str=None, # Please set this correctly if you want GPU
                ):
        #An image paramterized by a fourier features fed into an MLP
        #The possible output range of these images is between 0 and 1
        #In other words, no pixel will ever have a value <0 or >1
        
        super().__init__(height,width,num_channels)
        
        self.hidden_dim  =hidden_dim
        self.mapping_size=mapping_size
        self.scale       =scale
        self.device      =device or 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # The following objects do NOT have parameters, and are not changed while optimizing this class
        self.xy_grid =get_xy_grid(height,width,batch_size=1).to(device)
        self.feature_extractor=GaussianFourierFeatureTransform(2, mapping_size, scale)
        self.features=nn.Parameter(self.feature_extractor(self.xy_grid).to(self.device), requires_grad=False) # pre-compute this if we're regressing on images
        
        H=hidden_dim # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=num_channels  # Shorter variable names let us align the code better
        M=2*mapping_size
        self.model = nn.Sequential(
                nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, C, kernel_size=1),
                nn.Sigmoid(),
            ).to(self.device)
    
    #def project(self,uv_maps):
    #    #TODO: Check if this function works well...
    #    #Right now consider it untested
    #    assert len(uv_maps.shape)==(4), 'uv_maps should be BCHW'
    #    assert uv_maps.shape[1]==2, 'Should have two channels: u,v'
    #    return self.model(self.feature_extractor(uv_maps))
    
    def forward(self):
        # Return all the images we've learned
        output = self.model(self.features).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    
###############################
######## TEXTURE PACKS ########
###############################
    

class LearnableTexturePack(nn.Module):
    def __init__(self,
                 height      :int   ,
                 width       :int   ,
                 num_channels:int   ,
                 num_textures:int   ,
                 get_learnable_image):
        
        #This is an abstract class, and is meant to be subclassed before use
        #TODO: Inherit from some list class, such as nn.ModuleList. That way we can access learnable_images by indexing them from self...

        super().__init__()
        
        self.height      =height
        self.width       =width
        self.num_channels=num_channels
        self.num_textures=num_textures
        
        assert callable(get_learnable_image)
        
        learnable_images=[get_learnable_image() for _ in range(num_textures)]
        learnable_images=nn.ModuleList(learnable_images)
        self.learnable_images=learnable_images
    
    def as_numpy_images(self):
        return [x.as_numpy_image() for x in self.learnable_images]
       
    def forward(self):
        #Returns a tensor of size (NT, NC, H, W)
        #Where NT=self.num_textures, NC=self.num_channels, H=self.height, W=self.width
        
        output = torch.stack(tuple(x() for x in self.learnable_images))
        assert output.shape==(self.num_textures, self.num_channels, self.height, self.width), str("WTF? "+str(output.shape)+" IS NOT "+str((self.num_textures, self.num_channels, self.height, self.width)))
        
        return output

    def __len__(self):
        #Returns the number of textures in the texture pack
        return len(self.learnable_images)

    
class LearnableTexturePackRaster(LearnableTexturePack):
    def __init__(self,
                 height      :int=256,
                 width       :int=256,
                 num_channels:int=  3,
                 num_textures:int=  1):
        
        get_learnable_image = lambda: LearnableImageRaster(height      ,
                                                           width       ,
                                                           num_channels)
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        
class LearnableTexturePackMLP(LearnableTexturePack):
    def __init__(self,
                 height      :int=256 , 
                 width       :int=256 ,
                 num_channels:int=3   ,
                 hidden_dim  :int=256 ,
                 device      :str=None,
                 num_textures:int=1   ):
        
        get_learnable_image = lambda: LearnableImageMLP(height      ,
                                                        width       ,
                                                        num_channels,
                                                        hidden_dim  ,
                                                        device      )
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        self.hidden_dim  =hidden_dim
        self.device      =device              
        
        
class LearnableTexturePackFourier(LearnableTexturePack):
    def __init__(self,
                 height      :int=256 , 
                 width       :int=256 ,
                 num_channels:int=3   ,
                 hidden_dim  :int=256 ,
                 mapping_size:int=128 ,
                 scale       :int=10  ,
                 device      :str=None,
                 num_textures:int=1   ):
        
        get_learnable_image = lambda: LearnableImageFourier(height      ,
                                                            width       ,
                                                            num_channels,
                                                            hidden_dim  ,
                                                            mapping_size,
                                                            scale       ,
                                                            device      )
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        self.hidden_dim  =hidden_dim  
        self.mapping_size=mapping_size
        self.scale       =scale       
        self.device      =device      
