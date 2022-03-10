#NOTE: This tool is actually meant to be used in the dataset directories.
#However, because I want to save this code in the repo and the datasets aren't in the repo,
#I'm keeping this code here.

def get_id_to_path_dict(coco):
    #Output like {1: 'image1.png', 2: 'image2.png', ...}
    output={}
    images=coco['images']
    for image in images:
        assert isinstance(image,dict)
        path=image['path']
        path=get_file_name(path)
        output[image['id']]=path
    return output

def get_id_to_category_dict(coco):
    #Output like {1: 'apple', 2: 'soda', 3: 'rubiks', 4: 'garlic', 5: 'alphabet'}
    output={}
    categories=coco['categories']
    for category in categories:
        assert isinstance(category,dict)
        output[category['id']]=category['name']
    return output

def load_coco(json_path):
    if is_valid_url(json_path):
        import json
        return json.loads(curl(json_path))
    return load_json(json_path)

def get_segmentation_contours(coco):
    #Output like {'image.png': {'apple':[[x,y],[x,y]...], 'cube':[[x,y],[x,y]...], ...}, 'image2.png': {...}, ...}
    output={}
    id_to_path=get_id_to_path_dict(coco)
    id_to_cat =get_id_to_category_dict(coco)
    for annotation in coco['annotations']:
        image_path=id_to_path[annotation['image_id'   ]]
        cat_name  =id_to_cat [annotation['category_id']]
        contour   =annotation['segmentation'] # [x0,y0,x1,y1...]
        assert len(contour)==1 #I don't know why they do this...
        contour = contour[0]
        contour_x =contour[0::2]
        contour_y =contour[1::2]
        contour   =list(map(list,zip(contour_x,contour_y))) #[[x0,y0],[x1,y1],...]
        output.setdefault(image_path,{})[cat_name]=contour
    return output

def run_test():
    coco=load_coco('https://gist.githubusercontent.com/SqrtRyan/77dfdd47986cb65e7dd24e6e236f22e6/raw/611b4a9d42ed6c010280d9f4fadda49c9d6257ad/gistfile1.txt')

def get_segmentation_mask(coco,image_path,resolution):
    height,width=resolution

def get_coco_colors(coco):
    #Output like {'apple': (87, 203, 226), 'soda': (153, 106, 221), ...}
    output={}
    for category in coco['categories']:
        output[category['name']]=hex_color_to_tuple(category['color'])
    return output

def hex_color_to_tuple(hex_color:str):
    #EXAMPLE:
    #     >>> hex_color_to_tuple('#007FFF')
    #    ans = (0, 127, 255)
    assert len(hex_color)==len('#ABCDEF')
    hex_color=hex_color[1:]
    r=int(hex_color[0:2],16)
    g=int(hex_color[2:4],16)
    b=int(hex_color[4:6],16)
    return r,g,b

def get_segmentation_masks(coco,**colors):
    contours   =get_segmentation_contours(coco)
    coco_colors=get_coco_colors          (coco)

    #The colors provided as arguments override coco's colors
    coco_colors.update(colors)
    colors=coco_colors

    output_images=[]

    for image_path in contours:
        if not file_exists(image_path):
            print('Skipping',image_path,'because that file doesnt exist')
            continue

        image=load_image(image_path)
        image=as_rgb_image (image)
        image=as_byte_image(image)

        black=image*0
        for category in contours[image_path]:
            contour=contours[image_path][category]
            color  =colors              [category]
            black=cv_draw_contour(black,contour,color=color,antialias=False,fill=True)

        output_image=horizontally_concatenated_images(image,black)
        output_image=as_rgb_image(output_image)
        output_image=as_byte_image(output_image)
        output_images.append(output_image)

    return output_images

def get_segmentation_mask_duets(coco,background=(0,0,0),**colors):
    #Please cd into the folder containing the images before running this function

    contours   =get_segmentation_contours(coco)
    coco_colors=get_coco_colors          (coco)

    #The colors provided as arguments override coco's colors
    coco_colors.update(colors)
    colors=coco_colors

    for image_path in contours:
        if not file_exists(image_path):
            print('Skipping',image_path,'because that file doesnt exist')
            continue

        image=load_image(image_path)
        image=as_rgb_image (image)
        image=as_byte_image(image)

        black=image*0
        black[:,:]=background
        for category in contours[image_path]:
            contour=contours[image_path][category]
            color  =colors              [category]
            black=cv_draw_contour(black,contour,color=color,antialias=False,fill=True)

        output_image=horizontally_concatenated_images(image,black)
        output_image=as_rgb_image(output_image)
        output_image=as_byte_image(output_image)
        yield image_path,output_image

def get_segmentation_mask_alphas(coco,background=255,**label_values):
    #Encode label values into alpha channels
    assert all(isinstance(x,int) for x in label_values.values())
    assert all(0<=x<=255         for x in label_values.values())
    colors={x:(0,0,y) for x,y in label_values.items()}
    duets=get_segmentation_mask_duets(coco,background=background,**colors)
    for image_path,duet in duets:
        image,mask=split_tensor_into_regions(duet,1,2)
        alpha=mask[:,:,2]
        r,g,b=extract_rgb_channels(image)
        image=compose_rgba_image(r,g,b,alpha)
        image=as_byte_image(image)
        assert is_rgba_image(image)
        yield image_path,image

def create_segmentation_photos():
    #This is currently meant for the five_items dataset
    #Please move into the photos directory before running this function
    output_folder='LabeledAlphaPhotos'
    coco=load_coco('https://gist.githubusercontent.com/SqrtRyan/77dfdd47986cb65e7dd24e6e236f22e6/raw/611b4a9d42ed6c010280d9f4fadda49c9d6257ad/gistfile1.txt')
    outputs=get_segmentation_mask_alphas(coco,background=255,garlic=100,alphabet=0,rubiks=50,apple=150,soda=200)
    for image_path,image in outputs:
        image_path=get_file_name(image_path)
        image_path=with_file_extension(image_path,'png',replace=True) #Use PNG's: we need the alpha channel!
        output_path=path_join(output_folder,image_path)
        save_image(image,output_path)
        print('Wrote',output_path)

