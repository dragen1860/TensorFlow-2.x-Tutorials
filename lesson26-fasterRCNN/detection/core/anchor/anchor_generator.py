import tensorflow as tf
from detection.utils.misc import calc_img_shapes, calc_batch_padded_shape

class AnchorGenerator:
    """
    This class operate on padded iamge, eg. [1216, 1216]
    and generate scales*ratios number of anchor boxes for each point in
    padded image, with stride = feature_strides
    number of anchor = (1216 // feature_stride)^2
    number of anchor boxes = number of anchor * (scales_len*ratio_len)
    """
    def __init__(self, 
                 scales=(32, 64, 128, 256, 512), 
                 ratios=(0.5, 1, 2), 
                 feature_strides=(4, 8, 16, 32, 64)):
        '''
        Anchor Generator
        
        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        '''
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
     
    def generate_pyramid_anchors(self, img_metas):
        '''
        Generate the multi-level anchors for Region Proposal Network
        
        Args
        ---
            img_metas: [batch_size, 11]
        
        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''
        # generate anchors
        pad_shape = calc_batch_padded_shape(img_metas) # [1216, 1216]
        # <class 'list'>: [(304, 304), (152, 152), (76, 76), (38, 38), (19, 19)]
        feature_shapes = [(pad_shape[0] // stride, pad_shape[1] // stride)
                          for stride in self.feature_strides]
        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ] # [277248, 4], [69312, 4], [17328, 4], [4332, 4], [1083, 4]
        anchors = tf.concat(anchors, axis=0) # [369303, 4]
        # print('total anchors:', anchors.shape)
        # print('---------')

        # generate valid flags
        img_shapes = calc_img_shapes(img_metas) # (800, 1067)
        valid_flags = [
            self._generate_valid_flags(anchors, img_shapes[i])
            for i in range(img_shapes.shape[0])
        ]
        valid_flags = tf.stack(valid_flags, axis=0)
        
        anchors = tf.stop_gradient(anchors)
        valid_flags = tf.stop_gradient(valid_flags)
        
        return anchors, valid_flags
    
    def _generate_valid_flags(self, anchors, img_shape):
        '''
        remove these anchor boxed on padded area
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)
            
        Returns
        ---
            valid_flags: [num_anchors]
        '''
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2 # [369300]
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2
        
        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32) # [369300]
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        # set boxes whose center is out of image area as invalid.
        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)
        
        return valid_flags
    
    def _generate_level_anchors(self, level, feature_shape):
        '''Generate the anchors given the spatial shape of feature map.
        
        scale: 32
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (256, 256) (256, 256)
        scale: 64
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (128, 128) (128, 128)
        scale: 128
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (64, 64) (64, 64)
        scale: 256
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (32, 32) (32, 32)
        scale: 512
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (16, 16) (16, 16)


        scale: 32
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (304, 304) (304, 304)
        boxes: (277248, 4)
        scale: 64
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (152, 152) (152, 152)
        boxes: (69312, 4)
        scale: 128
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (76, 76) (76, 76)
        boxes: (17328, 4)
        scale: 256
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (38, 38) (38, 38)
        boxes: (4332, 4)
        scale: 512
        ratios: tf.Tensor([0.5 1.  2. ], shape=(3,), dtype=float32)
        pos: (19, 19) (19, 19)
        boxes: (1083, 4)
        total anchors: (369303, 4)

        ---
            feature_shape: (height, width)

        Returns
        ---
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]
        '''
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]
        
        # Get all combinations of scales and ratios
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1]) # [32, 32, 32]
        ratios = tf.reshape(ratios, [-1]) # [0.5, 1, 2]
        
        # Enumerate heights and widths from scales and ratios
        heights = scales / tf.sqrt(ratios) # [45, 32, 22], square root
        widths = scales * tf.sqrt(ratios)  # [22, 32, 45]

        # Enumerate shifts in feature space, [0, 4, ..., 1216-4]
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)
        
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y) # [304, 304, 2] coordinates

        # Enumerate combinations of shifts, widths, and heights # mesh A: [3] B:[304,304]=>[92416] =>[92416,3,2]
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

        # Convert to corner coordinates (y1, x1, y2, x2) [304x304, 3, 4] => [277448, 4]
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)
        # print('scale:', scale)
        # print('ratios:', ratios)
        # print('pos:', shifts_x.shape, shifts_y.shape)
        # print('boxes:', boxes.shape)
        return boxes
