import tensorflow as tf

def trim_zeros(boxes, name=None):
    '''
    Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def parse_image_meta(meta):
    '''
    Parses a tensor that contains image attributes to its components.
    
    Args
    ---
        meta: [..., 11]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    meta = meta.numpy()
    ori_shape = meta[..., 0:3]
    img_shape = meta[..., 3:6]
    pad_shape = meta[..., 6:9]
    scale = meta[..., 9]  
    flip = meta[..., 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }

def calc_batch_padded_shape(meta):
    '''
    Args
    ---
        meta: [batch_size, 11]
    
    Returns
    ---
        nd.ndarray. Tuple of (height, width)
    '''
    return tf.cast(tf.reduce_max(meta[:, 6:8], axis=0), tf.int32).numpy()

def calc_img_shapes(meta):
    '''
    Args
    ---
        meta: [..., 11]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def calc_pad_shapes(meta):
    '''
    Args
    ---
        meta: [..., 11]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    return tf.cast(meta[..., 6:8], tf.int32).numpy()