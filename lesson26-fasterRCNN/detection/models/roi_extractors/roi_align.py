import tensorflow as tf

from detection.utils.misc import *

class PyramidROIAlign(tf.keras.layers.Layer):

    def __init__(self, pool_shape, **kwargs):
        '''
        Implements ROI Pooling on multiple levels of the feature pyramid.

        Attributes
        ---
            pool_shape: (height, width) of the output pooled regions.
                Example: (7, 7)
        '''
        super(PyramidROIAlign, self).__init__(**kwargs)

        self.pool_shape = tuple(pool_shape)

    def call(self, inputs, training=True):
        '''
        Args
        ---
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in normalized coordinates.
            feature_map_list: List of [batch, height, width, channels].
                feature maps from different levels of the pyramid.
            img_metas: [batch_size, 11]

        Returns
        ---
            pooled_rois_list: list of [num_rois, pooled_height, pooled_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        '''
        rois_list, feature_map_list, img_metas = inputs

        pad_shapes = calc_pad_shapes(img_metas)
        
        pad_areas = pad_shapes[:, 0] * pad_shapes[:, 1]
        
        num_rois_list = [rois.shape.as_list()[0] for rois in rois_list]
        roi_indices = tf.constant(
            [i for i in range(len(rois_list)) for _ in range(rois_list[i].shape.as_list()[0])],
            dtype=tf.int32
        )
        
        areas = tf.constant(
            [pad_areas[i] for i in range(pad_areas.shape[0]) for _ in range(num_rois_list[i])],
            dtype=tf.float32
        )


        rois = tf.concat(rois_list, axis=0)
        
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(rois, 4, axis=1)
        h = y2 - y1
        w = x2 - x1
        
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4

        roi_level = tf.math.log(tf.sqrt(tf.squeeze(h * w, 1)) / tf.cast((224.0 / tf.sqrt(areas * 1.0)), tf.float32)) / tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))


        
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled_rois = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_rois = tf.gather_nd(rois, ix)

            # ROI indicies for crop_and_resize.
            level_roi_indices = tf.gather_nd(roi_indices, ix)

            # Keep track of which roi is mapped to which level
            roi_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_rois, pool_height, pool_width, channels]
            pooled_rois.append(tf.image.crop_and_resize(
                feature_map_list[i], level_rois, level_roi_indices, self.pool_shape,
                method="bilinear"))
            
        # Pack pooled features into one tensor
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # Pack roi_to_level mapping into one array and add another
        # column representing the order of pooled rois
        roi_to_level = tf.concat(roi_to_level, axis=0)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1)
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original rois
        # Sort roi_to_level by batch then roi index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            roi_to_level)[0]).indices[::-1]
        ix = tf.gather(roi_to_level[:, 1], ix)
        pooled_rois = tf.gather(pooled_rois, ix)
        
        pooled_rois_list = tf.split(pooled_rois, num_rois_list, axis=0)
        return pooled_rois_list
