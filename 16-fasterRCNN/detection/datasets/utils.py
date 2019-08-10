import cv2
import numpy as np

###########################################
#
# Utility Functions for 
# Image Preprocessing and Data Augmentation
#
###########################################

def img_flip(img):
    '''Flip the image horizontally
    
    Args
    ---
        img: [height, width, channel]
    
    Returns
    ---
        np.ndarray: the flipped image.
    '''
    return np.fliplr(img)

def bbox_flip(bboxes, img_shape):
    '''Flip bboxes horizontally.
    
    Args
    ---
        bboxes: [..., 4]
        img_shape: Tuple. (height, width)
    
    Returns
    ---
        np.ndarray: the flipped bboxes.
    '''
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 1] = w - bboxes[..., 3] - 1
    flipped[..., 3] = w - bboxes[..., 1] - 1
    return flipped

def impad_to_square(img, pad_size):
    '''Pad an image to ensure each edge to equal to pad_size.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded
        pad_size: Int.
    
    Returns
    ---
        ndarray: The padded image with shape of 
            [pad_size, pad_size, channels].
    '''
    shape = (pad_size, pad_size, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad

def impad_to_multiple(img, divisor):
    '''Pad an image to ensure each edge to be multiple to some number.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded.
        divisor: Int. Padded image edges will be multiple to divisor.
    
    Returns
    ---
        ndarray: The padded image.
    '''
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    shape = (pad_h, pad_w, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad

def imrescale(img, scale):
    '''Resize image while keeping the aspect ratio.
    
    Args
    ---
        img: [height, width, channels]. The input image.
        scale: Tuple of 2 integers. the image will be rescaled 
            as large as possible within the scale
    
    Returns
    ---
        np.ndarray: the scaled image.
    ''' 
    h, w = img.shape[:2]
    
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    
    new_size = (int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5))

    rescaled_img = cv2.resize(
        img, new_size, interpolation=cv2.INTER_LINEAR)
    
    return rescaled_img, scale_factor

def imnormalize(img, mean, std):
    '''Normalize the image.
    
    Args
    ---
        img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the normalized image.
    '''
    img = (img - mean) / std    
    return img.astype(np.float32)

def imdenormalize(norm_img, mean, std):
    '''Denormalize the image.
    
    Args
    ---
        norm_img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the denormalized image.
    '''
    img = norm_img * std + mean
    return img.astype(np.float32)

#######################################
#
# Utility Functions for Data Formatting
#
#######################################

def get_original_image(img, img_meta, 
                       mean=(0, 0, 0), std=(1, 1, 1)):
    '''Recover the origanal image.
    
    Args
    ---
        img: np.ndarray. [height, width, channel]. 
            The transformed image.
        img_meta: np.ndarray. [11]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the original image.
    '''
    img_meta_dict = parse_image_meta(img_meta)
    ori_shape = img_meta_dict['ori_shape']
    img_shape = img_meta_dict['img_shape']
    flip = img_meta_dict['flip']
    
    img = img[:img_shape[0], :img_shape[1]]
    if flip:
        img = img_flip(img)
    img = cv2.resize(img, (ori_shape[1], ori_shape[0]), 
                     interpolation=cv2.INTER_LINEAR)
    img = imdenormalize(img, mean, std)
    return img

def compose_image_meta(img_meta_dict):
    '''Takes attributes of an image and puts them in one 1D array.

    Args
    ---
        img_meta_dict: dict

    Returns
    ---
        img_meta: np.ndarray
    '''
    ori_shape = img_meta_dict['ori_shape']
    img_shape = img_meta_dict['img_shape']
    pad_shape = img_meta_dict['pad_shape']
    scale_factor = img_meta_dict['scale_factor']
    flip = 1 if img_meta_dict['flip'] else 0
    img_meta = np.array(
        ori_shape +               # size=3
        img_shape +               # size=3
        pad_shape +               # size=3
        tuple([scale_factor]) +   # size=1
        tuple([flip])             # size=1
    ).astype(np.float32)

    return img_meta

def parse_image_meta(img_meta):
    '''Parses an array that contains image attributes to its components.

    Args
    ---
        meta: [11]

    Returns
    ---
        a dict of the parsed values.
    '''
    ori_shape = img_meta[0:3]
    img_shape = img_meta[3:6]
    pad_shape = img_meta[6:9]
    scale_factor = img_meta[9]
    flip = img_meta[10]
    return {
        'ori_shape': ori_shape.astype(np.int32),
        'img_shape': img_shape.astype(np.int32),
        'pad_shape': pad_shape.astype(np.int32),
        'scale_factor': scale_factor.astype(np.float32),
        'flip': flip.astype(np.bool),
    }
