import torch


def condense_values(tensor,values):
    #Mutates the given tensor and returns it
    #   - tensor: can be either torch.Tensor or np.ndarray
    #   - values: should be a list of int
    #
    # EXAMPLES: Normal circumstances
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
    # EXAMPLES: When not all labels are covered
    #     >>> condense_values(torch.tensor([2,5,7,6]),[2,5,7,9])
    #    ans = tensor([0, 1, 2, 0])  # the last value is 0, because that's the default
    #
    # EXAMPLES: Works element-wise with arbitrary shapes
    #     >>> condense_values(torch.tensor([[2],[5],[7],[6]]),[2,5,7,9])
    #    ans = tensor([[0], [1], [2], [0]])
    #     >>> condense_values(torch.tensor([[2,5,7,6]]),[2,5,7,9])
    #    ans = tensor([[0, 1, 2, 0]])
    #
    # This function takes a 'tensor', and a list of numbers called 'values'
    # All elements in 'tensor' that are also in 'values' are simultaneously
    # replaced with their respective index in 'values'
    #
    # This function is used to let us specify which values of the b channel we're
    # using from the model images, so we can use multiple textures efficiently
    # See the uses of this function for a further understanding...

    for index,value in enumerate(values):
        tensor[tensor==value]=-index
    tensor[tensor>0]=0
    tensor=-tensor
    return tensor


def extract_scene_uvs_and_scene_labels(scene_images:torch.Tensor, label_values:list):
    #Takes an image obtained from an .exr file (with R,G,B) and return two torch tensors
    #It assumes all values are between [0,1] and that B encodes the label value while 
    #R,G represent the U,V values respectively
    #It also assumes that there are a maximum of 256 possible labels
    #
    #label_values is a list of label values, that will be condensed into the range [0,len(label_values)-1]

    #------- Inputs Validation -------
    
    assert len(scene_images.shape)==4, 'Images should be in the form BCHW'
    
    
    #------- Outputs Calculation -------
    
    r=scene_images[:,0,:,:]
    g=scene_images[:,1,:,:]
    b=scene_images[:,2,:,:]
    
    scene_labels=torch.floor(b*256).clamp(min=0,max=255).long()
    scene_labels=condense_values(scene_labels,label_values)
    
    u,v=r,g
    scene_uvs=torch.stack((u,v),dim=1)
    
    
    #------- Outputs Validation -------
    
    batch_size, num_channels, scene_height, scene_width = scene_images.shape
    
    assert scene_uvs   .shape == (batch_size, 2, scene_height, scene_width)
    assert scene_labels.shape == (batch_size,    scene_height, scene_width)
    
    assert scene_labels.max()<len(label_values) and scene_labels.min()>=0
    assert not scene_labels.dtype.is_floating_point
    
    return scene_uvs, scene_labels