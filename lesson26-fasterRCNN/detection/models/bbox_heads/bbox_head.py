import tensorflow as tf
from    tensorflow.keras import layers

from detection.core.bbox import transforms
from detection.core.loss import losses
from detection.utils.misc import *

class BBoxHead(tf.keras.Model):
    def __init__(self, num_classes, 
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.7,
                 nms_threshold=0.3,
                 max_instances=100,
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)
        
        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        
        self.rcnn_class_loss = losses.rcnn_class_loss
        self.rcnn_bbox_loss = losses.rcnn_bbox_loss
        
        self.rcnn_class_conv1 = layers.Conv2D(1024, self.pool_size, 
                                              padding='valid', name='rcnn_class_conv1')
        
        self.rcnn_class_bn1 = layers.BatchNormalization(name='rcnn_class_bn1')
        
        self.rcnn_class_conv2 = layers.Conv2D(1024, (1, 1), 
                                              name='rcnn_class_conv2')
        
        self.rcnn_class_bn2 = layers.BatchNormalization(name='rcnn_class_bn2')
        
        self.rcnn_class_logits = layers.Dense(num_classes, name='rcnn_class_logits')
        
        self.rcnn_delta_fc = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')
        
    def call(self, inputs, training=True):
        '''
        Args
        ---
            pooled_rois_list: List of [num_rois, pool_size, pool_size, channels]
        
        Returns
        ---
            rcnn_class_logits_list: List of [num_rois, num_classes]
            rcnn_probs_list: List of [num_rois, num_classes]
            rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        '''
        pooled_rois_list = inputs
        num_pooled_rois_list = [pooled_rois.shape[0] for pooled_rois in pooled_rois_list]
        pooled_rois = tf.concat(pooled_rois_list, axis=0)
        
        x = self.rcnn_class_conv1(pooled_rois)
        x = self.rcnn_class_bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.rcnn_class_conv2(x)
        x = self.rcnn_class_bn2(x, training=training)
        x = tf.nn.relu(x)
        
        x = tf.squeeze(tf.squeeze(x, 2), 1)
        
        logits = self.rcnn_class_logits(x)
        probs = tf.nn.softmax(logits)
        
        deltas = self.rcnn_delta_fc(x)
        deltas = tf.reshape(deltas, (-1, self.num_classes, 4))
        

        rcnn_class_logits_list = tf.split(logits, num_pooled_rois_list, 0)
        rcnn_probs_list = tf.split(probs, num_pooled_rois_list, 0)
        rcnn_deltas_list = tf.split(deltas, num_pooled_rois_list, 0)

            
        return rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list

    def loss(self, 
             rcnn_class_logits_list, rcnn_deltas_list, 
             rcnn_target_matchs_list, rcnn_target_deltas_list):
        """

        :param rcnn_class_logits_list:
        :param rcnn_deltas_list:
        :param rcnn_target_matchs_list:
        :param rcnn_target_deltas_list:
        :return:
        """
        rcnn_class_loss = self.rcnn_class_loss(
            rcnn_target_matchs_list, rcnn_class_logits_list)
        rcnn_bbox_loss = self.rcnn_bbox_loss(
            rcnn_target_deltas_list, rcnn_target_matchs_list, rcnn_deltas_list)
        
        return rcnn_class_loss, rcnn_bbox_loss
        
    def get_bboxes(self, rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas):
        '''
        Args
        ---
            rcnn_probs_list: List of [num_rois, num_classes]
            rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois_list: List of [num_rois, (y1, x1, y2, x2)]
            img_meta_list: [batch_size, 11]
        
        Returns
        ---
            detections_list: List of [num_detections, (y1, x1, y2, x2, class_id, score)]
                coordinates are in pixel coordinates.
        '''
        
        pad_shapes = calc_pad_shapes(img_metas)
        detections_list = [
            self._get_bboxes_single(
                rcnn_probs_list[i], rcnn_deltas_list[i], rois_list[i], pad_shapes[i])
            for i in range(img_metas.shape[0])
        ]
        return detections_list  
    
    def _get_bboxes_single(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        '''
        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: np.ndarray. [2]. (img_height, img_width)       
        '''
        H, W = img_shape   
        # Class IDs per ROI
        class_ids = tf.argmax(rcnn_probs, axis=1, output_type=tf.int32)
        
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(rcnn_probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(rcnn_probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(rcnn_deltas, indices)
        # Apply bounding box deltas
        # Shape: [num_rois, (y1, x1, y2, x2)] in normalized coordinates        
        refined_rois = transforms.delta2bbox(rois, deltas_specific, self.target_means, self.target_stds)
        
        # Clip boxes to image window
        refined_rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        window = tf.constant([0., 0., H * 1., W * 1.], dtype=tf.float32)
        refined_rois = transforms.bbox_clip(refined_rois, window)
        
        
        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        
        # Filter out low confidence boxes
        if self.min_confidence:
            conf_keep = tf.where(class_scores >= self.min_confidence)[:, 0]
            keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]
            
        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois,   keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            '''Apply Non-Maximum Suppression on ROIs of the given class.'''
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.max_instances,
                    iou_threshold=self.nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            return class_keep

        # 2. Map over class IDs
        nms_keep = []
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))
        nms_keep = tf.concat(nms_keep, axis=0)
        
        # 3. Compute intersection between keep and nms_keep
        keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        # Keep top detections
        roi_count = self.max_instances
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)  
        
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)
        
        return detections
        