import numpy as np
import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import *

class ProposalTarget:

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5):
        '''
        Compute regression and classification targets for proposals.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            num_rcnn_deltas: int. Maximal number of RoIs per image to feed to bbox heads.

        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
            
    def build_targets(self, proposals_list, gt_boxes, gt_class_ids, img_metas):
        '''
        Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.
        
        Args
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]
            
        Returns
        ---
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs_list: list of [num_rois]. Integer class IDs.
            rcnn_target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))].
            
        Note that self.num_rcnn_deltas >= num_rois > num_positive_rois. And different 
           images in one batch may have different num_rois and num_positive_rois.
        '''
        
        pad_shapes = calc_pad_shapes(img_metas) # [[1216, 1216]]
        
        rois_list = []
        rcnn_target_matchs_list = []
        rcnn_target_deltas_list = []
        
        for i in range(img_metas.shape[0]):
            rois, target_matchs, target_deltas = self._build_single_target(
                proposals_list[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i])
            rois_list.append(rois) # [192, 4], including pos/neg anchors
            rcnn_target_matchs_list.append(target_matchs) # positive target label, and padding with zero for neg
            rcnn_target_deltas_list.append(target_deltas) # positive target deltas, and padding with zero for neg
        
        return rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list
    
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape):
        '''
        Args
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            
        Returns
        ---
            rois: [num_rois, (y1, x1, y2, x2)]
            target_matchs: [num_positive_rois]
            target_deltas: [num_positive_rois, (dy, dx, log(dh), log(dw))]
        '''
        H, W = img_shape # 1216, 1216
        
        
        gt_boxes, non_zeros = trim_zeros(gt_boxes) # [7, 4], remove padded zero boxes
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros) # [7]
        # normalize (y1, x1, y2, x2) => 0~1
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        # [2k, 4] with [7, 4] => [2k, 7] overlop scores
        overlaps = geometry.compute_overlaps(proposals, gt_boxes)
        anchor_iou_argmax = tf.argmax(overlaps, axis=1) # [2000]get cloest gt boxed id for each anchor boxes
        roi_iou_max = tf.reduce_max(overlaps, axis=1) # [2000]get clost gt boxes overlop score for each anchor boxes
        # roi_iou_max: [2000],
        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr) #[2000]
        positive_indices = tf.where(positive_roi_bool)[:, 0] #[48, 1] =>[48]
        # get all positive indices, namely get all pos_anchor indices
        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]
        # get all negative anchor indices
        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.num_rcnn_deltas * self.positive_fraction) # 0.25?
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count] # [256*0.25]=64, at most get 64
        positive_count = tf.shape(positive_indices)[0] # 34
        
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.positive_fraction
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count #102
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count] #[102]
        
        # Gather selected ROIs, based on remove redundant pos/neg indices
        positive_rois = tf.gather(proposals, positive_indices) # [34, 4]
        negative_rois = tf.gather(proposals, negative_indices) # [102, 4]
        
        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices) # [34, 7]
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1) # [34]for each anchor, get its clost gt boxes
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment) # [34, 4]
        target_matchs = tf.gather(gt_class_ids, roi_gt_box_assignment) # [34]
        # target_matchs, target_deltas all get!!
        # proposal: [34, 4], target: [34, 4]
        target_deltas = transforms.bbox2delta(positive_rois, roi_gt_boxes, self.target_means, self.target_stds)
        # [34, 4] [102, 4]
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        
        N = tf.shape(negative_rois)[0] # 102
        target_matchs = tf.pad(target_matchs, [(0, N)]) # [34] padding after with [N]
        
        target_matchs = tf.stop_gradient(target_matchs) # [34+102]
        target_deltas = tf.stop_gradient(target_deltas) # [34, 4]
        # rois: [34+102, 4]
        return rois, target_matchs, target_deltas