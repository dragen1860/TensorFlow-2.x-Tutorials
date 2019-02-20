import tensorflow as tf
layers = tf.keras.layers

from detection.core.bbox import transforms
from detection.utils.misc import *

from detection.core.anchor import anchor_generator, anchor_target
from detection.core.loss import losses

class RPNHead(tf.keras.Model):
    def __init__(self, 
                 anchor_scales=(32, 64, 128, 256, 512), 
                 anchor_ratios=(0.5, 1, 2), 
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000, 
                 nms_threshold=0.7, 
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 **kwags):
        '''Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_feature_strides: Stride of the feature map relative 
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum 
                supression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        super(RPNHead, self).__init__(**kwags)
        
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales, 
            ratios=anchor_ratios, 
            feature_strides=anchor_feature_strides)
        
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=target_means, 
            target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)
        
        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss
        
        
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal', 
                                             name='rpn_conv_shared')
        
        self.rpn_class_raw = layers.Conv2D(2 * len(anchor_ratios), (1, 1),
                                           kernel_initializer='he_normal', 
                                           name='rpn_class_raw')

        self.rpn_delta_pred = layers.Conv2D(len(anchor_ratios) * 4, (1, 1),
                                           kernel_initializer='he_normal', 
                                           name='rpn_bbox_pred')
        
    def __call__(self, inputs, training=True):
        '''
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                one level of pyramid feat-maps.
        
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        
        layer_outputs = []
        
        for feat in inputs:
            shared = self.rpn_conv_shared(feat)
            shared = tf.nn.relu(shared)

            x = self.rpn_class_raw(shared)
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
            rpn_probs = tf.nn.softmax(rpn_class_logits)

            x = self.rpn_delta_pred(shared)
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
            
            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])

        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        
        return rpn_class_logits, rpn_probs, rpn_deltas

    def loss(self, rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas):
        '''Calculate rpn loss
        '''
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        
        rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
            anchors, valid_flags, gt_boxes, gt_class_ids)
        
        rpn_class_loss = self.rpn_class_loss(
            rpn_target_matchs, rpn_class_logits)
        rpn_bbox_loss = self.rpn_bbox_loss(
            rpn_target_deltas, rpn_target_matchs, rpn_deltas)
        
        return rpn_class_loss, rpn_bbox_loss
    
    def get_proposals(self, 
                      rpn_probs, 
                      rpn_deltas, 
                      img_metas, 
                      with_probs=False):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            img_metas: [batch_size, 11]
            with_probs: bool.
        
        Returns
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in 
                normalized coordinates if with_probs is False. 
                Otherwise, the shape of proposals in proposals_list is 
                [num_proposals, (y1, x1, y2, x2, score)]
        
        Note that num_proposals is no more than proposal_count. And different 
           images in one batch may have different num_proposals.
        '''
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        
        rpn_probs = rpn_probs[:, :, 1]
        
        pad_shapes = calc_pad_shapes(img_metas)
        
        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shapes[i], with_probs)
            for i in range(img_metas.shape[0])
        ]
        
        return proposals_list
    
    def _get_proposals_single(self, 
                              rpn_probs, 
                              rpn_deltas, 
                              anchors, 
                              valid_flags, 
                              img_shape, 
                              with_probs):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
                pixel coordinates.
            valid_flags: [num_anchors]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            with_probs: bool.
        
        Returns
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in normalized 
                coordinates.
        '''
        
        H, W = img_shape
        
        # filter invalid anchors
        valid_flags = tf.cast(valid_flags, tf.bool)
        
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)

        # Improve performance
        pre_nms_limit = min(6000, anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)
        
        # Get refined anchors
        proposals = transforms.delta2bbox(anchors, rpn_deltas, 
                                          self.target_means, self.target_stds)
        
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = transforms.bbox_clip(proposals, window)
        
        # Normalize
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # NMS
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, indices)
        
        if with_probs:
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            proposals = tf.concat([proposals, proposal_probs], axis=1)
   
        return proposals
        
        